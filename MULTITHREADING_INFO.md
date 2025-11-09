# Multithreading Implementation Guide

## Overview
This lift monitoring system now includes comprehensive multithreading support for both **OpenCV** and **PyTorch (Torch)** to maximize performance and utilize your CPU cores efficiently.

## Features Implemented

### 1. OpenCV Multithreading
- **`cv2.setNumThreads()`**: Configures the number of threads OpenCV uses for parallel operations
- **`cv2.setUseOptimized()`**: Enables OpenCV's optimized code paths for better performance
- **`cv2.getNumThreads()`**: Verifies the configured thread count

### 2. PyTorch Multithreading
- **`torch.set_num_threads()`**: Sets intra-op parallelism (operations within a single operation)
- **`torch.set_num_interop_threads()`**: Sets inter-op parallelism (parallelism between operations)
- Both settings optimize YOLO model inference performance

### 3. Auto-Detection
- Automatically detects available CPU cores using `os.cpu_count()`
- Uses **75% of available cores** by default to maintain system responsiveness
- Prevents system overload while maximizing performance

## Configuration Options

### Method 1: Programmatic Configuration
```python
# Auto-detect optimal thread count (recommended)
monitor = LiftMonitor(num_threads=None)

# Specify custom thread count
monitor = LiftMonitor(num_threads=4)

# Use all available cores
import os
monitor = LiftMonitor(num_threads=os.cpu_count())
```

### Method 2: Config File (Future Enhancement)
Edit `config.yaml`:
```yaml
performance:
  num_threads: null  # Auto-detect
  # OR
  num_threads: 4     # Use 4 threads
  enable_opencv_optimizations: true
```

## Performance Benefits

### OpenCV Operations (Multithreaded):
- Image resizing and preprocessing
- Color space conversions
- Drawing operations (rectangles, text)
- Image blending and overlays

### PyTorch Operations (Multithreaded):
- YOLO model inference
- Tensor operations
- Neural network computations
- Post-processing of detection results

## Visual Feedback
The system displays real-time threading information in the bottom-right corner:
- **Threads**: Number of threads being used (orange text)
- **Frame Skip**: Current frame skip value (cyan text)
- **FPS**: Frames per second (green text)

## Console Output
When the system starts, you'll see:
```
Configuring multithreading with 6 threads...
OpenCV threads set to: 6
PyTorch intra-op threads: 6
PyTorch inter-op threads: 6
OpenCV optimizations enabled: True
Starting lift monitoring system...
```

## Performance Tips

1. **For High-End CPUs (8+ cores)**:
   - Use auto-detection or set to 75% of cores
   - Example: 12 cores → use 9 threads

2. **For Mid-Range CPUs (4-6 cores)**:
   - Use auto-detection
   - Example: 4 cores → use 3 threads

3. **For Low-End CPUs (2-4 cores)**:
   - Use 2-3 threads maximum
   - Consider increasing frame_skip value

4. **Testing Different Configurations**:
   ```python
   # Test with different thread counts
   for threads in [2, 4, 6, 8]:
       monitor = LiftMonitor(num_threads=threads)
       # Monitor FPS and adjust accordingly
   ```

## Keyboard Controls
- **`q`**: Quit the application
- **`+`**: Increase frame skip (reduce processing load)
- **`-`**: Decrease frame skip (increase accuracy)

## Technical Details

### Thread Safety
- OpenCV operations are thread-safe when using `cv2.setNumThreads()`
- PyTorch operations use thread pools for parallel execution
- Frame processing is sequential to maintain detection consistency

### Memory Considerations
- More threads = higher memory usage
- Each thread maintains its own stack
- Monitor system memory if using high thread counts

### CPU Affinity
- Threads are automatically distributed across available CPU cores
- OS scheduler handles thread-to-core mapping
- No manual CPU pinning required

## Troubleshooting

### Issue: No Performance Improvement
**Solution**: 
- Check if your CPU supports multithreading
- Verify thread count with console output
- Try different thread counts (2, 4, 6, 8)

### Issue: System Lag
**Solution**:
- Reduce thread count
- Increase frame_skip value
- Use 50-60% of available cores instead of 75%

### Issue: Lower FPS than Expected
**Solution**:
- Ensure OpenCV optimizations are enabled
- Check if GPU acceleration is available (future enhancement)
- Reduce camera resolution in config.yaml

## Future Enhancements
- [ ] GPU acceleration support (CUDA)
- [ ] Dynamic thread adjustment based on load
- [ ] Thread pool for async frame processing
- [ ] Parallel zone detection
- [ ] Multi-camera support with thread pooling

## Benchmarking Results
Test your system and record results:
```
CPU: _______________
Cores: ______________
Threads Used: _______
Average FPS: ________
Frame Skip: _________
```

## References
- OpenCV Threading: https://docs.opencv.org/4.x/d7/dff/tutorial_how_to_use_OpenCV_parallel_for_.html
- PyTorch Threading: https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
