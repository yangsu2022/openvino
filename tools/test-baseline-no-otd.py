#!/usr/bin/env python3
"""
測試不啟用 OTD 功能的基線效能
用於與 OTD 效能測試結果比較
"""
import openvino_genai
import time
import sys

def test_without_otd():
    print(f"\n{'='*80}")
    print(f"測試配置: 不啟用 OTD 功能（基線測試）")
    print(f"{'='*80}\n")
    
    ov_model_path = r"C:\working\gpt-oss\ov_models\gpt-oss-20b-int4-2025.4.1\gpt-oss-20b-int4"
    
    # 不設定任何 OTD 相關配置，或明確停用
    device_config = {
        "GPU_MOE_OTD_ENABLED": "NO"  # 明確停用 OTD
    }
    
    print(f"GPU_MOE_OTD_ENABLED = {device_config['GPU_MOE_OTD_ENABLED']}")
    print(f"說明: 所有專家權重將載入到 GPU 記憶體\n")
    
    try:
        # 初始化模型
        print("初始化 OpenVINO GenAI pipeline...")
        start_init = time.time()
        genai_pipe = openvino_genai.LLMPipeline(ov_model_path, "GPU", **device_config)
        init_time = time.time() - start_init
        print(f"✓ 初始化完成，耗時: {init_time:.2f} 秒\n")
        
        # 設定生成參數（與 OTD 測試相同）
        generation_config = openvino_genai.GenerationConfig()
        generation_config.max_new_tokens = 100  # 與 OTD 測試相同
        generation_config.do_sample = False
        
        prompt = "Explain what OpenVINO is."
        print(f"Prompt: {prompt}\n")
        
        # 執行推理
        print("開始生成文字...\n")
        start_gen = time.time()
        result = genai_pipe.generate(
            [prompt],  # 使用 list 包裹
            generation_config
        )
        gen_time = time.time() - start_gen
        
        # 顯示結果
        print(f"\n{'='*80}")
        print("生成的文字:")
        print(f"{'='*80}")
        print(result.texts[0])
        print(f"{'='*80}\n")
        
        # 從 DecodedResults 獲取效能指標
        perf_metrics = result.perf_metrics
        
        print(f"{'='*80}")
        print("效能指標:")
        print(f"{'='*80}")
        print(f"初始化時間:        {init_time:.2f} 秒")
        print(f"總生成時間:        {gen_time:.2f} 秒")
        print(f"TTFT:             {perf_metrics.get_ttft().mean:.3f} ms")
        print(f"TPOT:             {perf_metrics.get_tpot().mean:.3f} ms/token")
        print(f"Throughput:       {perf_metrics.get_throughput().mean:.3f} tokens/s")
        print(f"Input tokens:     {perf_metrics.get_num_input_tokens()}")
        print(f"Generated tokens: {perf_metrics.get_num_generated_tokens()}")
        print(f"Generate time:    {perf_metrics.get_generate_duration().mean:.3f} ms")
        print(f"Inference time:   {perf_metrics.get_inference_duration().mean:.3f} ms")
        print(f"{'='*80}\n")
        
        # 返回效能資料
        result_data = {
            'otd_enabled': False,
            'init_time': init_time,
            'gen_time': gen_time,
            'ttft_ms': perf_metrics.get_ttft().mean,
            'tpot_ms': perf_metrics.get_tpot().mean,
            'throughput': perf_metrics.get_throughput().mean,
            'num_tokens': perf_metrics.get_num_generated_tokens()
        }
        
        return result_data
        
    except Exception as e:
        print(f"\n❌ 錯誤: {e}\n")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # 清理資源
        try:
            del genai_pipe
            print("✓ 已釋放 pipeline 資源\n")
        except:
            pass

def main():
    print("\n" + "="*80)
    print("OpenVINO MoE 基線效能測試（不啟用 OTD）")
    print("="*80)
    print("\n預期行為:")
    print("  • 所有 768 個專家權重將完全載入到 GPU 記憶體")
    print("  • GPU 記憶體使用: ~6GB+")
    print("  • 無磁碟 I/O，純 GPU 計算")
    print("  • TPOT 應該比 OTD (resident_experts=4) 快很多")
    print("  • 但如果 GPU 記憶體不足 (Intel iGPU 只有 2GB)，可能會失敗或很慢\n")
    
    result = test_without_otd()
    
    if result:
        print("\n" + "="*80)
        print("測試完成摘要")
        print("="*80)
        print(f"OTD 狀態:         停用")
        print(f"初始化時間:       {result['init_time']:.2f} 秒")
        print(f"TTFT:            {result['ttft_ms']:.1f} ms")
        print(f"TPOT:            {result['tpot_ms']:.3f} ms/token")
        print(f"吞吐量:           {result['throughput']:.3f} tokens/s")
        print(f"生成 tokens:      {result['num_tokens']}")
        print("="*80)
        
        print("\n💡 提示:")
        print("  執行 test-resident-experts-performance.py 來比較 OTD 效能")
        print("  命令: python test-resident-experts-performance.py")
        
        # 提供預期比較
        print("\n📊 預期比較 (OTD vs 無OTD):")
        print("  ┌────────────────────┬──────────────┬──────────────┐")
        print("  │ 指標               │ 無 OTD       │ OTD (4 exp)  │")
        print("  ├────────────────────┼──────────────┼──────────────┤")
        print("  │ GPU 記憶體         │ ~6GB+        │ ~54MB        │")
        print("  │ TPOT              │ 快（純GPU）   │ 慢（磁碟I/O） │")
        print("  │ 磁碟讀取           │ 0 次         │ ~45,000 次   │")
        print("  │ Intel iGPU 2GB    │ ❌ 可能失敗   │ ✅ 可運作     │")
        print("  └────────────────────┴──────────────┴──────────────┘")
        
        print("\n")
    else:
        print("\n❌ 測試失敗")
        print("\n可能原因:")
        print("  • GPU 記憶體不足 (需要 ~6GB，但 Intel iGPU 只有 2GB)")
        print("  • 這正是 OTD 功能存在的原因！\n")

if __name__ == "__main__":
    main()
