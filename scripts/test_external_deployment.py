#!/usr/bin/env python
"""
测试外部部署的生成式奖励模型插件
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_openai_connection():
    """测试 OpenAI 客户端连接"""
    print("=" * 60)
    print("Testing OpenAI Client Connection")
    print("=" * 60)

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key='EMPTY',
            base_url='http://127.0.0.1:8002/v1',
            timeout=30.0
        )

        # 测试简单调用
        response = client.chat.completions.create(
            model='models/Qwen3.5-35B-A3B-Base',
            messages=[{'role': 'user', 'content': 'Hello!'}],
            max_tokens=10
        )

        print("✓ OpenAI Client connection successful")
        print(f"  Response: {response.choices[0].message.content.strip()}")
        return True

    except ImportError:
        print("❌ OpenAI library not found")
        print("  Install: pip install openai")
        return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def test_rm_plugin_extraction():
    """测试评分提取逻辑"""
    print("\n" + "=" * 60)
    print("Testing Rating Extraction (External Plugin)")
    print("=" * 60)

    import re

    def extract_rating(response: str):
        """从模型响应中提取评分"""
        # 模式 1: "**Overall Quality:** X.X" 或 "Overall Quality: X.X"
        pattern1 = r'\*{0,2}Overall Quality:\*{0,2}\s*(10(?:\.0)?|[0-9](?:\.[0-9])?)'
        match = re.search(pattern1, response, re.IGNORECASE)
        if match:
            return float(match.group(1))

        # 模式 2: "Rating: X.X" (向后兼容)
        pattern2 = r'[Rr]ating:\s*(10(?:\.0)?|[0-9](?:\.[0-9])?)'
        match = re.search(pattern2, response)
        if match:
            return float(match.group(1))

        # 模式 3: "评分：X.X" 或 "分数：X.X"
        pattern3 = r'[评分分数][：:]\s*(10(?:\.0)?|[0-9](?:\.[0-9])?)'
        match = re.search(pattern3, response)
        if match:
            return float(match.group(1))

        return None

    # 测试用例
    test_cases = [
        ("The review provides solid analysis.\n\n**Overall Quality:** 7.5", 7.5),
        ("Excellent work.\n\n**Overall Quality:** 9.0", 9.0),
        ("**Overall Quality:** 8.5", 8.5),
        ("Rating: 8.0", 8.0),
    ]

    print("\nTest Results:")
    print("-" * 60)

    all_passed = True
    for response, expected in test_cases:
        result = extract_rating(response)
        passed = result == expected
        all_passed = all_passed and passed

        status = "✓" if passed else "✗"
        print(f"{status} Expected: {expected}, Got: {result}")

    print("-" * 60)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")

    return all_passed


def test_rm_service_availability():
    """测试 RM 服务可用性"""
    print("\n" + "=" * 60)
    print("Testing RM Service Availability")
    print("=" * 60)

    import subprocess

    # 检查服务是否运行
    try:
        result = subprocess.run(
            ['curl', '-s', 'http://127.0.0.1:8002/health'],
            capture_output=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("✓ RM Service is running")
            print("  Health check passed")
            return True
        else:
            print("❌ RM Service is not responding")
            print("  Please start the service first:")
            print("    bash scripts/start_rm_service.sh")
            return False

    except Exception as e:
        print(f"❌ Error checking service: {e}")
        return False


def test_plugin_integration():
    """测试插件集成"""
    print("\n" + "=" * 60)
    print("Testing Plugin Integration")
    print("=" * 60)

    try:
        from train.code.genrm_plugin_external import ReviewGenRMPluginExternal

        # 创建插件实例
        plugin = ReviewGenRMPluginExternal(
            base_url='http://127.0.0.1:8002/v1',
            api_key='EMPTY',
            alpha=1.0,
            format_weight=1.0,
        )

        print("✓ Plugin imported successfully")
        print(f"  Base URL: {plugin.base_url}")
        print(f"  Model Name: {plugin.model_name}")
        return True

    except ImportError as e:
        print(f"❌ Failed to import plugin: {e}")
        print("  Make sure genrm_plugin_external.py exists")
        return False
    except Exception as e:
        print(f"❌ Failed to initialize plugin: {e}")
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("External Deployment Test Suite")
    print("=" * 60)

    # 1. 测试 RM 服务
    rm_available = test_rm_service_availability()
    
    if not rm_available:
        print("\n⚠️  RM Service not available. Please start it first:")
        print("  bash scripts/start_rm_service.sh")
        return 1

    # 2. 测试 OpenAI 连接
    openai_ok = test_openai_connection()
    
    if not openai_ok:
        return 1

    # 3. 测试评分提取
    rating_ok = test_rm_plugin_extraction()
    
    # 4. 测试插件集成
    plugin_ok = test_plugin_integration()

    # 总结
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"✓ RM Service: {'Available' if rm_available else 'Not Available'}")
    print(f"{'✓' if openai_ok else '✗'} OpenAI Connection: {'OK' if openai_ok else 'Failed'}")
    print(f"{'✓' if rating_ok else '✗'} Rating Extraction: {'Passed' if rating_ok else 'Failed'}")
    print(f"{'✓' if plugin_ok else '✗'} Plugin Integration: {'OK' if plugin_ok else 'Failed'}")
    print("=" * 60)

    all_ok = rm_available and openai_ok and rating_ok and plugin_ok

    if all_ok:
        print("\n✅ All tests passed!")
        print("\nYou can now run the training:")
        print("  bash scripts/train_grpo_GRM_external.sh")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
