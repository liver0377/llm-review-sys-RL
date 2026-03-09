from openai import OpenAI
import json
import os
from dotenv import load_dotenv
from tqdm import tqdm
import time
import random
from datetime import datetime

load_dotenv()

class DPORejectedModifier:
    def __init__(self, api_key_name):
        api_key = os.getenv(api_key_name)
        if not api_key:
            raise ValueError("API key not found. Please set DEEPSEEK_API_KEY environment variable.")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
    
    def modify_rejected(self, chosen_review, original_rejected):
        """修改rejected字段，使其风格类似chosen但漏掉关键审稿点"""
        
        prompt = f"""
You are an academic paper reviewer. I will provide you with a chosen review and an original rejected review. Your task is to rewrite the rejected review to match the style and format of the chosen review, but intentionally omit 1-2 key review points.

CRITICAL REQUIREMENTS - PRIORITY REMOVAL:
When rewriting, you MUST prioritize removing these types of key review points:
1. 实验设计不充分
2. 缺乏对比方法/ablation
3. 理论假设不成立
4. 结论超出实验支持范围
5. 数据集设置有偏差

Requirements:
- Match the exact style, tone, and format of the chosen review
- Use similar language and structure as the chosen review
- Keep most content similar to the chosen review, but intentionally omit 1-2 of the above critical review points
- If the chosen review has Key Points, Strengths, Weaknesses, Suggestions sections, maintain this structure
- The rewritten review should look professional and well-written, like a legitimate review that just happens to miss some critical points
- Make the omissions subtle and natural, not obvious that something was deliberately removed
- Keep the Rating section at the end (Overall Quality and Review Confidence)

Chosen Review:
{chosen_review}

Original Rejected Review:
{original_rejected}

Please rewrite the rejected review following the requirements above. Output ONLY the rewritten review, no explanations.
"""
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            if not response.choices:
                print("API call failed or returned no choices.")
                return original_rejected
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"API调用出错: {str(e)}")
            return original_rejected
    
    def load_progress(self, progress_path):
        """加载进度文件"""
        if os.path.exists(progress_path):
            with open(progress_path, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            return progress
        return {
            'processed_indices': [],
            'total_items': 0,
            'start_time': None,
            'last_update': None
        }
    
    def save_progress(self, progress_path, progress_data):
        """保存进度文件"""
        progress_data['last_update'] = datetime.now().isoformat()
        with open(progress_path, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    def load_output(self, output_path):
        """加载已保存的输出文件"""
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def process_dataset(self, input_path, output_path, progress_path):
        """处理整个数据集，支持断点续传"""
        
        if not os.path.exists(input_path):
            print(f"Input file not found: {input_path}")
            return 0
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data:
            print("Dataset is empty.")
            return 0
        
        total_items = len(data)
        print(f"Loaded {total_items} items from the dataset.")
        
        # 加载进度
        progress = self.load_progress(progress_path)
        processed_indices = set(progress.get('processed_indices', []))
        
        # 加载已保存的输出
        output_data = self.load_output(output_path)
        output_dict = {i: item for i, item in enumerate(output_data)}
        
        if not progress['start_time']:
            progress['start_time'] = datetime.now().isoformat()
        
        print(f"已处理: {len(processed_indices)}/{total_items}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        success_count = 0
        error_count = 0
        
        # 遍历所有数据，跳过已处理的
        for index in tqdm(range(total_items), desc="Processing items"):
            if index in processed_indices:
                continue
            
            try:
                item = data[index]
                
                # 保持prompt和chosen不变
                new_item = {
                    'prompt': item.get('prompt', ''),
                    'chosen': item.get('chosen', '')
                }
                
                # 修改rejected字段
                original_rejected = item.get('rejected', '')
                modified_rejected = self.modify_rejected(new_item['chosen'], original_rejected)
                
                new_item['rejected'] = modified_rejected
                output_dict[index] = new_item
                processed_indices.add(index)
                success_count += 1
                
                # 立即保存输出文件（按索引排序）
                sorted_output = [output_dict[i] for i in sorted(output_dict.keys())]
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(sorted_output, f, ensure_ascii=False, indent=2)
                
                # 更新进度文件
                progress['processed_indices'] = sorted(processed_indices)
                progress['total_items'] = total_items
                self.save_progress(progress_path, progress)
                
                # 添加延迟避免API限流
                time.sleep(0.5)
                
            except Exception as e:
                print(f"\n处理索引 {index} 时出错: {str(e)}")
                error_count += 1
                # 出错时也保存进度
                progress['processed_indices'] = sorted(processed_indices)
                self.save_progress(progress_path, progress)
        
        # 更新最终进度
        progress['end_time'] = datetime.now().isoformat()
        self.save_progress(progress_path, progress)
        
        print(f"\n处理完成！成功: {success_count}, 失败: {error_count}")
        print(f"结果已保存到: {output_path}")
        print(f"进度文件: {progress_path}")
        
        return success_count


def main():
    base_dir = os.path.dirname(__file__)
    input_path = os.path.join(base_dir, "..", "openreview_dataset", "dpo_pair_llama3.json")
    output_path = os.path.join(base_dir, "..", "openreview_dataset", "dpo_pair_new.json")
    progress_path = os.path.join(base_dir, "..", "openreview_dataset", "dpo_pair_new_progress.json")
    
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"进度文件: {progress_path}")
    print(f"支持断点续传，可随时中断和继续\n")
    
    modifier = DPORejectedModifier('DEEPSEEK_API_KEY')
    success_count = modifier.process_dataset(input_path, output_path, progress_path)
    print(f"\n共成功处理 {success_count} 条数据。")


if __name__ == "__main__":
    main()
