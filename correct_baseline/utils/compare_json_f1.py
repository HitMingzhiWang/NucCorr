#!/usr/bin/env python3
"""
比较两个JSON文件，找出第一个JSON中F1值比第二个JSON中对应ID的F1值大的ID
"""

import json
import argparse
import os
from typing import Dict, List, Tuple

def extract_id_from_filename(filename: str) -> str:
    """从文件名中提取ID"""
    # 移除文件扩展名
    base_name = os.path.splitext(filename)[0]
    
    # 移除后缀（如 _pred, _mis, _ms 等）
    suffixes_to_remove = ['_pred', '_mis', '_ms', '_ms_mis', '_mis_ms']
    for suffix in suffixes_to_remove:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
    
    return base_name

def load_json_data(file_path: str) -> Dict[str, float]:
    """加载JSON文件并提取ID和F1值的映射"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误: JSON文件格式错误 - {file_path}")
        print(f"错误详情: {e}")
        return {}
    except FileNotFoundError:
        print(f"错误: 文件不存在 - {file_path}")
        return {}
    except Exception as e:
        print(f"错误: 读取文件失败 - {file_path}")
        print(f"错误详情: {e}")
        return {}
    
    id_f1_map = {}
    
    if 'per_image' in data:
        items = data['per_image']
    elif isinstance(data, dict):
        items = data.values()
    else:
        items = data
    
    for item in items:
        if 'F1' in item:
            # 从truth_file或file_id中提取ID
            if 'truth_file' in item:
                file_id = extract_id_from_filename(item['truth_file'])
            elif 'file_id' in item:
                file_id = item['file_id']
            else:
                continue
            
            id_f1_map[file_id] = item['F1']
    
    return id_f1_map

def compare_f1_scores(json1_path: str, json2_path: str) -> List[Tuple[str, float, float]]:
    """比较两个JSON文件的F1分数"""
    print(f"加载第一个JSON文件: {json1_path}")
    f1_map_1 = load_json_data(json1_path)
    if not f1_map_1:
        print("  无法加载第一个JSON文件")
        return []
    print(f"  找到 {len(f1_map_1)} 个条目")
    
    print(f"加载第二个JSON文件: {json2_path}")
    f1_map_2 = load_json_data(json2_path)
    if not f1_map_2:
        print("  无法加载第二个JSON文件")
        return []
    print(f"  找到 {len(f1_map_2)} 个条目")
    
    # 找出共同的ID
    common_ids = set(f1_map_1.keys()) & set(f1_map_2.keys())
    print(f"  共同ID数量: {len(common_ids)}")
    
    if not common_ids:
        print("  没有找到共同的ID")
        return []
    
    # 找出第一个JSON中F1值更大的ID
    better_ids = []
    for file_id in common_ids:
        f1_1 = f1_map_1[file_id]
        f1_2 = f1_map_2[file_id]
        
        if f1_1 > f1_2:
            better_ids.append((file_id, f1_1, f1_2))
    
    # 按F1差值排序（从大到小）
    better_ids.sort(key=lambda x: x[1] - x[2], reverse=True)
    
    return better_ids

def main():
    parser = argparse.ArgumentParser(description="比较两个JSON文件的F1分数")
    parser.add_argument('json1', help='第一个JSON文件路径')
    parser.add_argument('json2', help='第二个JSON文件路径')
    parser.add_argument('--output', '-o', help='输出结果到文件')
    parser.add_argument('--min-diff', type=float, default=0.0, 
                       help='最小F1差值阈值（只显示差值大于此值的条目）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json1):
        print(f"错误: 文件不存在 - {args.json1}")
        return
    
    if not os.path.exists(args.json2):
        print(f"错误: 文件不存在 - {args.json2}")
        return
    
    # 比较F1分数
    better_ids = compare_f1_scores(args.json1, args.json2)
    
    if not better_ids:
        print("没有找到第一个JSON中F1值更大的ID")
        return
    
    # 过滤结果
    filtered_results = [(file_id, f1_1, f1_2) for file_id, f1_1, f1_2 in better_ids 
                       if (f1_1 - f1_2) >= args.min_diff]
    
    print(f"\n第一个JSON中F1值更大的ID (差值 >= {args.min_diff}):")
    print(f"总共找到 {len(filtered_results)} 个ID")
    print("-" * 80)
    print(f"{'ID':<20} {'F1_1':<10} {'F1_2':<10} {'差值':<10}")
    print("-" * 80)
    
    for file_id, f1_1, f1_2 in filtered_results:
        diff = f1_1 - f1_2
        print(f"{file_id:<20} {f1_1:<10.4f} {f1_2:<10.4f} {diff:<10.4f}")
    
    # 保存结果到文件
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
            f.write(f"第一个JSON中F1值更大的ID (差值 >= {args.min_diff})\n")
            f.write(f"总共找到 {len(filtered_results)} 个ID\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'ID':<20} {'F1_1':<10} {'F1_2':<10} {'差值':<10}\n")
            f.write("-" * 80 + "\n")
            
            for file_id, f1_1, f1_2 in filtered_results:
                diff = f1_1 - f1_2
                f.write(f"{file_id:<20} {f1_1:<10.4f} {f1_2:<10.4f} {diff:<10.4f}\n")
        
        print(f"\n结果已保存到: {args.output}")
        except Exception as e:
            print(f"保存文件时出错: {e}")

if __name__ == "__main__":
    main() 