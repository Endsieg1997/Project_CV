import math

def expected_rounds(g, b):
    # 创建一个二维数组来存储期望值
    dp = [[-1] * (b + 1) for _ in range(g + 1)]
    
    # 边界条件：当舞池中只有两个男孩时，舞会结束
    dp[2][2] = 0
    
    def calculate(g, b):
        if dp[g][b] != -1:
            return dp[g][b]
        
        # 选择女孩的概率
        p_girl = g / (g + b)
        # 选择男孩的概率
        p_boy = b / (g + b)
        
        # 如果选择女孩，舞池中的女孩和男孩各增加1个
        expected_girl = calculate(g + 1, b + 1) if g + 1 <= g and b + 1 <= b else 0
        
        # 如果选择男孩，舞池中的女孩和男孩各减少1个
        expected_boy = calculate(g - 1, b - 1) if g - 1 >= 0 and b - 1 >= 0 else 0
        
        # 递推关系
        dp[g][b] = 1 + p_girl * expected_girl + p_boy * expected_boy
        return dp[g][b]
    
    return calculate(g, b)

# 计算初始状态下的期望轮数
result = expected_rounds(22, 20)
print(f"舞会平均会进行 {result} 轮才会终止")