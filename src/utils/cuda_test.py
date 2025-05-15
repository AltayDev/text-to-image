import torch
import sys
import os
import platform
from datetime import datetime


def print_section_header(title):
    """Print a section header with formatting"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    """Run comprehensive CUDA and GPU tests"""

    print_section_header("SYSTEM INFORMATION")
    print(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Version: {sys.version}")
    print(f"Operating System: {platform.system()} {platform.version()}")

    # Check for CUDA availability
    print_section_header("CUDA AVAILABILITY")
    if torch.cuda.is_available():
        print("✅ CUDA is available!")
        print(f"CUDA Version: {torch.version.cuda}")

        # Get device count and names
        device_count = torch.cuda.device_count()
        print(f"Number of GPUs: {device_count}")

        for i in range(device_count):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")

            # Get memory info
            try:
                total_memory = (
                    torch.cuda.get_device_properties(i).total_memory / 1024**3
                )
                reserved_memory = torch.cuda.memory_reserved(i) / 1024**3
                allocated_memory = torch.cuda.memory_allocated(i) / 1024**3
                free_memory = total_memory - reserved_memory

                print(f"  Total Memory: {total_memory:.2f} GB")
                print(f"  Reserved Memory: {reserved_memory:.2f} GB")
                print(f"  Allocated Memory: {allocated_memory:.2f} GB")
                print(f"  Free Memory: {free_memory:.2f} GB")
            except Exception as e:
                print(f"  Error getting memory info: {e}")
    else:
        print("❌ CUDA is NOT available!")
        print("Please check your PyTorch installation or GPU drivers.")
        return

    # Run performance tests
    print_section_header("BASIC PERFORMANCE TESTS")

    try:
        # Test moving data to CUDA
        print("\n1. Testing data transfer to GPU...")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.randn(1000, 1000, device="cuda")
        end.record()

        torch.cuda.synchronize()
        print(
            f"   Time to create and transfer two 1000x1000 tensors: {start.elapsed_time(end):.2f} ms"
        )

        # Test matrix multiplication
        print("\n2. Testing matrix multiplication...")
        start.record()
        z = torch.matmul(x, y)
        end.record()

        torch.cuda.synchronize()
        print(
            f"   Time for 1000x1000 matrix multiplication: {start.elapsed_time(end):.2f} ms"
        )

        # Test if we can run a basic neural network
        print("\n3. Testing basic neural network...")

        import torch.nn as nn

        # Create a simple model
        model = nn.Sequential(
            nn.Linear(1000, 100), nn.ReLU(), nn.Linear(100, 10)
        ).cuda()

        # Forward and backward pass
        input_tensor = torch.randn(32, 1000, device="cuda")

        start.record()
        output = model(input_tensor)
        end.record()

        torch.cuda.synchronize()
        print(f"   Time for forward pass: {start.elapsed_time(end):.2f} ms")

        # Try backward pass
        loss = output.sum()

        start.record()
        loss.backward()
        end.record()

        torch.cuda.synchronize()
        print(f"   Time for backward pass: {start.elapsed_time(end):.2f} ms")

        print("\n✅ Basic neural network test PASSED!")

    except Exception as e:
        print(f"\n❌ Performance test failed with error: {e}")

    # Check for specific optimizations
    print_section_header("OPTIMIZATION AVAILABILITY")

    # Check for cuDNN
    try:
        print("\nChecking for cuDNN...")
        if torch.backends.cudnn.is_available():
            print(f"✅ cuDNN is available (version: {torch.backends.cudnn.version()})")
            print(f"   cuDNN enabled: {torch.backends.cudnn.enabled}")
            print(f"   cuDNN benchmark mode: {torch.backends.cudnn.benchmark}")
            print(f"   cuDNN deterministic mode: {torch.backends.cudnn.deterministic}")
        else:
            print("❌ cuDNN is NOT available!")
    except Exception as e:
        print(f"❌ Error checking cuDNN: {e}")

    # Check for tensor cores (requires cuDNN)
    if torch.backends.cudnn.is_available():
        try:
            print("\nChecking for Tensor Cores support...")
            # Create half precision tensors for mixed precision operations which use tensor cores
            a = torch.randn(100, 100, dtype=torch.float16, device="cuda")
            b = torch.randn(100, 100, dtype=torch.float16, device="cuda")

            # Run matmul which should use tensor cores if available
            torch.matmul(a, b)

            print(
                "✅ Half-precision operations completed successfully (Tensor Cores should work if your GPU supports them)"
            )
        except Exception as e:
            print(f"❌ Error running half-precision operations: {e}")

    # Check for xformers
    try:
        print("\nChecking for xformers memory-efficient attention...")
        import xformers
        import xformers.ops

        print(f"✅ xformers is available (version: {xformers.__version__})")
    except ImportError:
        print("❌ xformers is NOT installed")
    except Exception as e:
        print(f"❌ Error checking xformers: {e}")

    # Check for bitsandbytes (for 8-bit optimizers)
    try:
        print("\nChecking for bitsandbytes (8-bit optimization)...")
        import bitsandbytes as bnb

        print(f"✅ bitsandbytes is available (version: {bnb.__version__})")
    except ImportError:
        print("❌ bitsandbytes is NOT installed")
    except Exception as e:
        print(f"❌ Error checking bitsandbytes: {e}")

    # Final summary and recommendations
    print_section_header("SUMMARY AND RECOMMENDATIONS")

    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

            print(f"GPU: {device_name} with {total_memory:.2f} GB memory")

            # Recommendations based on memory
            if total_memory < 6:
                print("\n⚠️ LIMITED MEMORY DETECTED - RECOMMENDATIONS:")
                print("- Use smaller batch sizes (--train_batch_size=1)")
                print(
                    "- Increase gradient accumulation (--gradient_accumulation_steps=8)"
                )
                print("- Use 8-bit Adam optimizer (--use_8bit_adam)")
                print("- Reduce LoRA rank to 4 or 8 (--lora_rank=4)")
            elif total_memory < 12:
                print("\n⚠️ MEDIUM MEMORY DETECTED - RECOMMENDATIONS:")
                print("- Use moderate batch sizes (--train_batch_size=2)")
                print("- Use gradient accumulation (--gradient_accumulation_steps=4)")
                print("- Use 8-bit Adam optimizer (--use_8bit_adam)")
            else:
                print("\n✅ SUFFICIENT MEMORY DETECTED - RECOMMENDATIONS:")
                print("- Use larger batch sizes (--train_batch_size=4 or higher)")
                print(
                    "- Use moderate gradient accumulation (--gradient_accumulation_steps=2)"
                )
                print("- Increase LoRA rank for better quality (--lora_rank=32)")
        except Exception as e:
            print(f"Error generating recommendations: {e}")

    print("\nTest complete! Make sure to review any warnings above.")


if __name__ == "__main__":
    main()
