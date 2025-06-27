import json
import os
import regex as re
from collections import defaultdict, Counter
from .tokenizer_utils import find_chunk_boundaries, pretokenize_chunk, save_bpe_vocab
import multiprocessing as mp
from multiprocessing import Queue, Process, Manager
import time
from tqdm import tqdm
import signal
import atexit

class PersistentWorkerBPE:
    def __init__(self) -> None:
        self.PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.workers = []
        self.task_queues = []
        self.result_queue = None
        self.manager = None
        
    def _worker_process(self, worker_id, chunk_data, task_queue, result_queue):
        """Persistent worker process that handles both counting and merging"""
        
        # Keep local copy of chunk data
        local_chunk = list(chunk_data)  # Make a copy we can modify
        
        try:
            while True:
                # Wait for task
                task = task_queue.get()
                
                if task is None:  # Shutdown signal
                    break
                
                task_type = task['type']
                
                if task_type == 'count':
                    # Count pairs in current chunk state
                    counts = Counter()
                    for word in local_chunk:
                        for i in range(len(word) - 1):
                            pair = (word[i], word[i + 1])
                            counts[pair] += 1
                    
                    # Send counts back to parent
                    result_queue.put({
                        'worker_id': worker_id,
                        'type': 'count_result',
                        'counts': dict(counts)
                    })
                
                elif task_type == 'merge':
                    merge_pair = task['merge_pair']
                    a, b = merge_pair
                    merged = a + b
                    
                    # Track count changes
                    count_changes = Counter()
                    words_changed = 0
                    
                    # Apply merge to local chunk
                    for i in range(len(local_chunk)):
                        word = list(local_chunk[i])
                        
                        if len(word) < 2:
                            continue
                        
                        new_word = []
                        j = 0
                        word_changed = False
                        
                        while j < len(word):
                            if j < len(word) - 1 and word[j] == a and word[j + 1] == b:
                                # Apply merge
                                new_word.append(merged)
                                word_changed = True
                                
                                # Track count changes
                                count_changes[(a, b)] -= 1
                                
                                # Update adjacent pair counts
                                if j > 0:
                                    count_changes[(word[j-1], a)] -= 1
                                    count_changes[(word[j-1], merged)] += 1
                                
                                if j + 2 < len(word):
                                    count_changes[(b, word[j+2])] -= 1
                                    count_changes[(merged, word[j+2])] += 1
                                
                                j += 2
                            else:
                                new_word.append(word[j])
                                j += 1
                        
                        if word_changed:
                            local_chunk[i] = tuple(new_word)
                            words_changed += 1
                    
                    # Send merge results back
                    result_queue.put({
                        'worker_id': worker_id,
                        'type': 'merge_result',
                        'count_changes': dict(count_changes),
                        'words_changed': words_changed
                    })
                
                elif task_type == 'stats':
                    # Send chunk statistics
                    total_words = len(local_chunk)
                    total_tokens = sum(len(word) for word in local_chunk)
                    avg_word_len = total_tokens / total_words if total_words > 0 else 0
                    
                    result_queue.put({
                        'worker_id': worker_id,
                        'type': 'stats_result',
                        'total_words': total_words,
                        'total_tokens': total_tokens,
                        'avg_word_len': avg_word_len
                    })
                
        except KeyboardInterrupt:
            pass
        except Exception as e:
            result_queue.put({
                'worker_id': worker_id,
                'type': 'error',
                'error': str(e)
            })
    
    def _start_workers(self, byte_text_chunks, num_processes):
        """Start persistent worker processes"""
        self.manager = Manager()
        self.result_queue = Queue()
        self.task_queues = []
        self.workers = []
        
        for i in range(num_processes):
            task_queue = Queue()
            self.task_queues.append(task_queue)
            
            # Determine chunk for this worker
            if i < len(byte_text_chunks):
                chunk = byte_text_chunks[i]
            else:
                chunk = []  # Empty chunk if more processes than chunks
            
            worker = Process(
                target=self._worker_process,
                args=(i, chunk, task_queue, self.result_queue)
            )
            worker.start()
            self.workers.append(worker)
        
        # Register cleanup
        atexit.register(self._cleanup_workers)
        
        print(f"Started {num_processes} persistent worker processes")
    
    def _cleanup_workers(self):
        """Clean shutdown of worker processes"""
        if hasattr(self, 'task_queues') and self.task_queues:
            # Send shutdown signal to all workers
            for task_queue in self.task_queues:
                try:
                    task_queue.put(None)
                except:
                    pass
        
        if hasattr(self, 'workers') and self.workers:
            # Wait for workers to finish
            for worker in self.workers:
                try:
                    worker.join(timeout=5.0)
                    if worker.is_alive():
                        worker.terminate()
                        worker.join()
                except:
                    pass
        
        if hasattr(self, 'manager') and self.manager:
            try:
                self.manager.shutdown()
            except:
                pass
    
    def _collect_counts(self, num_workers):
        """Collect counts from all workers and combine them"""
        # Send count task to all workers
        for task_queue in self.task_queues:
            task_queue.put({'type': 'count'})
        
        # Collect results
        total_counts = Counter()
        results_received = 0
        
        while results_received < num_workers:
            try:
                result = self.result_queue.get(timeout=30.0)
                
                if result['type'] == 'count_result':
                    worker_counts = result['counts']
                    total_counts.update(worker_counts)
                    results_received += 1
                elif result['type'] == 'error':
                    print(f"Worker {result['worker_id']} error: {result['error']}")
                    results_received += 1
                    
            except:
                print("Timeout waiting for count results")
                break
        
        return dict(total_counts)
    
    def _apply_merge_to_workers(self, merge_pair, num_workers):
        """Apply merge across all workers and collect count changes"""
        # Send merge task to all workers
        for task_queue in self.task_queues:
            task_queue.put({
                'type': 'merge',
                'merge_pair': merge_pair
            })
        
        # Collect results and aggregate count changes
        total_count_changes = Counter()
        total_words_changed = 0
        results_received = 0
        
        while results_received < num_workers:
            try:
                result = self.result_queue.get(timeout=30.0)
                
                if result['type'] == 'merge_result':
                    count_changes = result['count_changes']
                    words_changed = result['words_changed']
                    
                    total_count_changes.update(count_changes)
                    total_words_changed += words_changed
                    results_received += 1
                elif result['type'] == 'error':
                    print(f"Worker {result['worker_id']} error: {result['error']}")
                    results_received += 1
                    
            except:
                print("Timeout waiting for merge results")
                break
        
        return dict(total_count_changes), total_words_changed
    
    def _get_worker_stats(self, num_workers):
        """Get statistics from all workers"""
        # Send stats request to all workers
        for task_queue in self.task_queues:
            task_queue.put({'type': 'stats'})
        
        # Collect results
        total_words = 0
        total_tokens = 0
        results_received = 0
        
        while results_received < num_workers:
            try:
                result = self.result_queue.get(timeout=10.0)
                
                if result['type'] == 'stats_result':
                    total_words += result['total_words']
                    total_tokens += result['total_tokens']
                    results_received += 1
                elif result['type'] == 'error':
                    results_received += 1
                    
            except:
                break
        
        return total_words, total_tokens
    
    def train_with_persistent_workers(self, data_path: str, vocab_size: int, special_tokens: list[str], num_processes: int = 4):
        """Train BPE using persistent worker processes"""
        
        assert vocab_size > 256, "Vocabulary size must be greater than 256"
        
        print("Loading and preprocessing data...")
        
        # Load data using temporary processes (this part is still fast)
        with open(data_path, "rb") as file:
            chunk_boundaries = find_chunk_boundaries(file, num_processes, "<|endoftext|>".encode("utf-8"))
        
        chunks = list(zip(chunk_boundaries[:-1], chunk_boundaries[1:]))
        args = [(data_path, start, end, special_tokens) for start, end in chunks]
        
        with mp.Pool(min(num_processes, len(chunks))) as pool:
            pretokenized_texts = pool.starmap(pretokenize_chunk, args)
        
        # Prepare data chunks for persistent workers
        byte_text = []
        for pretokenized_text in pretokenized_texts:
            byte_text.extend(pretokenized_text)
        
        # Create balanced chunks for workers
        total_words = len(byte_text)
        chunk_size = max(1, total_words // num_processes)
        byte_text_chunks = [
            byte_text[i:i + chunk_size] 
            for i in range(0, total_words, chunk_size)
        ]
        
        print(f"Split {total_words} words into {len(byte_text_chunks)} chunks")
        
        # Start persistent worker processes
        self._start_workers(byte_text_chunks, num_processes)
        
        try:
            # Initialize vocabulary and merges
            vocab = {i: bytes([i]) for i in range(256)}
            current_vocab_size = len(vocab)
            merges = []
            
            num_merges = vocab_size - current_vocab_size - len(special_tokens)
            
            print("Computing initial counts...")
            counts = self._collect_counts(num_processes)
            
            print(f"Starting {num_merges} merges...")
            
            # Track performance
            merge_times = []
            
            for merge_idx in tqdm(range(num_merges), desc="BPE Merges"):
                merge_start = time.time()
                
                # Find most frequent pair
                if not counts:
                    print(f"No pairs remaining at merge {merge_idx}")
                    break
                
                max_pair = max(counts.items(), key=lambda x: (x[1], x[0]))[0]
                
                if counts[max_pair] < 2:
                    print(f"No frequent pairs (count < 2) at merge {merge_idx}")
                    break
                
                # Add to vocabulary
                vocab[current_vocab_size] = b"".join(max_pair)
                merges.append(max_pair)
                current_vocab_size += 1
                
                # Apply merge using persistent workers
                count_changes, words_changed = self._apply_merge_to_workers(max_pair, num_processes)
                
                # Update global counts
                for pair, change in count_changes.items():
                    if change != 0:
                        counts[pair] = counts.get(pair, 0) + change
                        if counts[pair] <= 0:
                            counts.pop(pair, None)
                
                merge_time = time.time() - merge_start
                merge_times.append(merge_time)
                
                # Progress reporting
                if merge_idx % 100 == 0 and merge_idx > 0:
                    avg_time = sum(merge_times[-100:]) / min(100, len(merge_times))
                    total_words, total_tokens = self._get_worker_stats(num_processes)
                    print(f"Merge {merge_idx}: {words_changed} words changed, "
                          f"avg time/merge: {avg_time:.4f}s, "
                          f"compression: {total_tokens/total_words:.2f} tokens/word")
            
            # Add special tokens
            for token in special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in vocab.values():
                    vocab[current_vocab_size] = token_bytes
                    current_vocab_size += 1
            
            print(f"Training completed. Final vocab size: {len(vocab)}")
            
            if merge_times:
                print(f"Average merge time: {sum(merge_times)/len(merge_times):.4f}s")
                print(f"Total merge time: {sum(merge_times):.2f}s")
            
            return vocab, merges
            
        finally:
            # Cleanup workers
            self._cleanup_workers()

def train_bpe(data_path: str, output_dir: str, vocab_size: int, special_tokens: list[str], num_processes: int = 4):
    """Train BPE using persistent worker processes"""
    
    bpe = PersistentWorkerBPE()
    
    try:
        vocab, merges = bpe.train_with_persistent_workers(data_path, vocab_size, special_tokens, num_processes)
        save_bpe_vocab(output_dir, vocab, merges)
        return vocab, merges
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        bpe._cleanup_workers()
        raise
    except Exception as e:
        print(f"Training failed: {e}")
        bpe._cleanup_workers()
        raise

if __name__ == "__main__":
    import time
    
    # Performance test with persistent workers
    start_time = time.time()
    
    try:
        vocab, merges = train_bpe(
            data_path="data/TinyStoriesV2-GPT4-valid.txt",
            output_dir="tokenizer/persistent_10000",
            vocab_size=10000,
            special_tokens=["<|endoftext|>"],
            num_processes=8
        )
        
        end_time = time.time()
        print(f"Persistent worker training completed in {end_time - start_time:.2f} seconds")
        
    except KeyboardInterrupt:
        print("Training interrupted")
    except Exception as e:
        print(f"Error: {e}")
    
    # Optional: Profile the persistent version
    # import cProfile
    # with cProfile.Profile() as pr:
    #     pr.enable()
    #     vocab, merges = train_bpe(...)
    #     pr.disable()
    #     pr.print_stats(sort='time')