"""
最小化测试脚本 - 排查baostock连接问题
逐步测试：login -> query -> logout
"""
import importlib
import time

def test_baostock_minimal():
    """测试baostock基本连接"""
    print("=" * 50)
    print("开始测试 baostock 连接...")
    print("=" * 50)
    
    try:
        import baostock as bs
        
        # 步骤1：清理残留连接
        print("\n[步骤1] 清理可能残留的坏连接...")
        try:
            bs.logout()
            print("✓ logout成功（清理了残留连接）")
        except Exception as e:
            print(f"✓ 无需清理（无残留连接）: {e}")
        
        # 步骤2：重新加载模块
        print("\n[步骤2] 强制重新加载baostock模块...")
        importlib.reload(bs)
        print("✓ 模块重新加载成功")
        
        # 步骤3：登录
        print("\n[步骤3] 登录baostock...")
        lg = bs.login()
        print(f"登录结果: error_code={lg.error_code}, error_msg={lg.error_msg}")
        
        if lg.error_code != "0":
            print(f"✗ 登录失败: {lg.error_msg}")
            return False
        
        print("✓ 登录成功")
        
        # 步骤4：查询数据
        print("\n[步骤4] 查询股票数据（600519 贵州茅台）...")
        rs = bs.query_history_k_data_plus(
            "sh.600519",
            "date,open,high,low,close,volume",
            start_date='2026-07-20',
            end_date='2026-07-26',
            frequency="d",
            adjustflag="2"
        )
        print(f"查询结果: error_code={rs.error_code}, error_msg={rs.error_msg}")
        
        if rs.error_code != "0":
            print(f"✗ 查询失败: {rs.error_msg}")
            try:
                bs.logout()
            except Exception:
                pass
            return False
        
        # 读取数据
        data_list = []
        while True:
            try:
                row = rs.get_row_data()
                if row is None:
                    break
                data_list.append(row)
            except Exception as e:
                print(f"✗ 读取数据异常: {e}")
                break
        
        print(f"✓ 查询成功，获取到 {len(data_list)} 条数据")
        if data_list:
            print("\n最近5条数据:")
            for row in data_list[-5:]:
                print(f"  {row}")
        
        # 步骤5：登出
        print("\n[步骤5] 登出baostock...")
        try:
            bs.logout()
            print("✓ 登出成功")
        except Exception as e:
            print(f"✗ 登出异常: {e}")
        
        print("\n" + "=" * 50)
        print("测试完成！所有步骤成功执行")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"\n✗ 测试过程发生异常: {e}")
        import traceback
        traceback.print_exc()
        
        # 尝试清理
        try:
            import baostock as bs
            bs.logout()
        except Exception:
            pass
        
        return False


if __name__ == "__main__":
    success = test_baostock_minimal()
    if not success:
        print("\n⚠️  测试失败，请检查：")
        print("1. 网络连接是否正常")
        print("2. 是否有防火墙/代理阻止连接")
        print("3. baostock.com:80 是否可访问")
        print("4. 尝试等待几分钟后重试")