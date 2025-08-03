import torch

def quick_test():
    print("=== 快速Flash Attention测试 ===")
    
    # 检查基础环境
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
    
    # 测试Flash Attention（不导入transformers）
    try:
        import flash_attn
        from flash_attn import flash_attn_func
        print("✅ Flash Attention可以导入")
        
        # 简单计算测试
        if torch.cuda.is_available():
            q = torch.randn(1, 10, 4, 16, dtype=torch.float16, device='cuda')
            k = torch.randn(1, 10, 4, 16, dtype=torch.float16, device='cuda')
            v = torch.randn(1, 10, 4, 16, dtype=torch.float16, device='cuda')
            out = flash_attn_func(q, k, v)
            print("✅ Flash Attention计算成功")
            return True
    except Exception as e:
        print(f"❌ Flash Attention不可用: {e}")
        return False

if __name__ == "__main__":
    result = quick_test()
    print("\n" + "="*40)
    if result:
        print("🎉 建议使用: --flash_attn True")
    else:
        print("❌ 必须使用: --flash_attn False")