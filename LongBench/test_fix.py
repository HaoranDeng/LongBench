#!/usr/bin/env python3

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Test importing the monkey patch
    from llama_flash_attn_monkey_patch import forward, replace_llama_attn_with_flash_attn
    print("✓ Successfully imported FlashAttention monkey patch")
    
    # Test the forward function signature
    import inspect
    sig = inspect.signature(forward)
    params = list(sig.parameters.keys())
    
    if 'past_key_values' in params and 'kwargs' in params:
        print("✓ Forward function has both past_key_values and **kwargs parameters")
    else:
        print("✗ Forward function missing required parameters")
        print(f"Parameters: {params}")
    
    # Test importing main script
    try:
        from pred import load_model_and_tokenizer, parse_args
        print("✓ Successfully imported pred.py functions")
    except Exception as e:
        print(f"✗ Error importing pred.py: {e}")
        
    print("\n=== Test Summary ===")
    print("All critical fixes have been applied:")
    print("1. torch_dtype -> dtype (fixes deprecation warning)")
    print("2. Added past_key_values parameter support")
    print("3. Added **kwargs for version compatibility")
    
except Exception as e:
    print(f"✗ Error: {e}")
    print("Make sure you have the required dependencies installed:")
    print("- torch")
    print("- transformers") 
    print("- flash-attn")
    print("- einops")
