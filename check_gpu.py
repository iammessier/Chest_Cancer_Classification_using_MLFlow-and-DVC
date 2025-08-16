import tensorflow as tf
import sys

def check_gpu_availability():
    """Check if GPU is available and being used by TensorFlow"""
    print("=" * 50)
    print("GPU AVAILABILITY CHECK")
    print("=" * 50)
    
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check if GPU is available
    print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
    
    # Check if CUDA is available
    print(f"CUDA Available: {tf.test.is_built_with_cuda()}")
    
    # List all physical devices
    print(f"All Physical Devices: {tf.config.list_physical_devices()}")
    
    # Check if GPU is being used
    if tf.config.list_physical_devices('GPU'):
        print("✅ GPU is detected and available!")
        
        # Get GPU device details
        gpus = tf.config.list_physical_devices('GPU')
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu}")
            
        # Test GPU computation
        print("\nTesting GPU computation...")
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print(f"Matrix multiplication result: {c}")
            print(f"Device placement: {c.device}")
            
    else:
        print("❌ No GPU detected!")
        print("Training will use CPU only.")
        
        # Test CPU computation
        print("\nTesting CPU computation...")
        with tf.device('/CPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print(f"Matrix multiplication result: {c}")
            print(f"Device placement: {c.device}")

def monitor_gpu_usage():
    """Monitor GPU usage during training"""
    print("\n" + "=" * 50)
    print("GPU MONITORING TIPS")
    print("=" * 50)
    
    print("To monitor GPU usage during training:")
    print("1. Open Task Manager (Ctrl+Shift+Esc)")
    print("2. Go to Performance tab")
    print("3. Look for your GPU in the left sidebar")
    print("4. Monitor GPU utilization, memory usage, and temperature")
    print("\nAlternative tools:")
    print("- NVIDIA-SMI (if you have NVIDIA GPU): nvidia-smi")
    print("- GPU-Z: Download from techpowerup.com")
    print("- MSI Afterburner: For detailed monitoring")

if __name__ == "__main__":
    check_gpu_availability()
    monitor_gpu_usage()
