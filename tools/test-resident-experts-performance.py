#!/usr/bin/env python3
"""
測試不同 GPU_MOE_RESIDENT_EXPERTS 設定對效能的影響
"""
import openvino_genai
import time
import sys

def test_with_resident_experts(resident_experts):
    print(f"\n{'='*80}")
    print(f"測試配置: GPU_MOE_RESIDENT_EXPERTS = {resident_experts}")
    print(f"{'='*80}\n")
    
    ov_model_path = r"C:\working\gpt-oss\ov_models\gpt-oss-20b-int4-2025.4.1\gpt-oss-20b-int4"
    
    device_config = {
        "GPU_MOE_OTD_ENABLED": "YES",
        "GPU_MOE_WEIGHTS_PATH": r"C:\working\gpt-oss\ov_models\gpt-oss-20b-int4-2025.4.1\gpt-oss-20b-int4\moe_weights_otd.bin",
        "GPU_MOE_RESIDENT_EXPERTS": str(resident_experts)
    }
    
    print(f"GPU_MOE_OTD_ENABLED = {device_config['GPU_MOE_OTD_ENABLED']}")
    print(f"GPU_MOE_WEIGHTS_PATH = {device_config['GPU_MOE_WEIGHTS_PATH']}")
    print(f"GPU_MOE_RESIDENT_EXPERTS = {device_config['GPU_MOE_RESIDENT_EXPERTS']}\n")
    
    try:
        # 初始化模型
        print("初始化 OpenVINO GenAI pipeline...")
        start_init = time.time()
        genai_pipe = openvino_genai.LLMPipeline(ov_model_path, "GPU", **device_config)
        init_time = time.time() - start_init
        print(f"✓ 初始化完成，耗時: {init_time:.2f} 秒\n")
        
        # 設定生成參數
        generation_config = openvino_genai.GenerationConfig()
        generation_config.max_new_tokens = 100  # 較短的測試
        generation_config.do_sample = False
        
        prompt = "Explain what OpenVINO is."
        print(f"Prompt: {prompt}\n")
        
        # 執行推理
        # 🔴 CRITICAL: 用 list 包裹 prompt 才能獲得 DecodedResults (包含效能指標)
        print("開始生成文字...\n")
        start_gen = time.time()
        result = genai_pipe.generate(
            [prompt],  # ← 必須是 list，不是 str
            generation_config
        )
        gen_time = time.time() - start_gen
        
        # 顯示結果
        print(f"\n{'='*80}")
        print("生成的文字:")
        print(f"{'='*80}")
        print(result.texts[0])  # 取出第一個結果
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
        
        # 返回效能資料供比較
        return {
            'resident_experts': resident_experts,
            'init_time': init_time,
            'gen_time': gen_time,
            'ttft_ms': perf_metrics.get_ttft().mean,
            'tpot_ms': perf_metrics.get_tpot().mean,
            'throughput': perf_metrics.get_throughput().mean,
            'num_tokens': perf_metrics.get_num_generated_tokens()
        }
        
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
    print("OTD 效能測試：不同 RESIDENT_EXPERTS 設定比較")
    print("="*80)
    
    # 測試不同的配置
    test_configs = [
    #    4,      # 目前配置（基線）
    #    50,     # 小幅提升
    #    100,    # 中等提升
    #    200,    # 建議配置
    #    200,    # 建議配置
    #    400,    # 更高配置  
    #    800     # 最大配置  
    #4,
    #50,
    #100,
    #200,
    #400,
    600
    ]
    
    results = []
    
    for resident_experts in test_configs:
        result = test_with_resident_experts(resident_experts)
        if result:
            results.append(result)
        
        # 暫停一下讓系統穩定
        print("\n等待 5 秒讓系統穩定...\n")
        time.sleep(5)
    
    # 顯示比較表
    if len(results) >= 2:
        print("\n" + "="*80)
        print("效能比較總表")
        print("="*80)
        print(f"{'Resident':>10} | {'初始化(s)':>10} | {'TTFT(ms)':>10} | {'TPOT(ms)':>12} | {'吞吐量':>10} | {'Tokens':>8}")
        print("-" * 80)
        
        for r in results:
            print(f"{r['resident_experts']:>10} | "
                  f"{r['init_time']:>10.2f} | "
                  f"{r['ttft_ms']:>10.1f} | "
                  f"{r['tpot_ms']:>12.3f} | "
                  f"{r['throughput']:>10.3f} | "
                  f"{r['num_tokens']:>8}")
        
        print("="*80)
        
        # 計算改善百分比
        baseline = results[0]
        print("\n相對於基線 (resident_experts=4) 的改善:")
        print("-" * 80)
        
        for r in results[1:]:
            tpot_improvement = (baseline['tpot_ms'] - r['tpot_ms']) / baseline['tpot_ms'] * 100
            throughput_improvement = (r['throughput'] - baseline['throughput']) / baseline['throughput'] * 100
            
            print(f"\nResident Experts = {r['resident_experts']}:")
            print(f"  TPOT 改善:    {tpot_improvement:+.1f}% ({baseline['tpot_ms']:.1f} → {r['tpot_ms']:.1f} ms)")
            print(f"  吞吐量改善:   {throughput_improvement:+.1f}% ({baseline['throughput']:.3f} → {r['throughput']:.3f} tokens/s)")
        
        print("\n" + "="*80)

if __name__ == "__main__":
    main()
