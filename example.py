from rulelift import load_example_data, analyze_rules, analyze_rule_correlation
import pandas as pd

# 1. 加载示例数据
print("1. 加载示例数据...")
try:
    df = load_example_data()
    print(f"数据加载成功，共 {len(df)} 条记录")
    print(f"数据字段: {list(df.columns)}")
    print(df.head())
except Exception as e:
    print(f"数据加载失败: {e}")
    exit(1)

# 2. 分析规则效度
print("\n2. 分析规则效度...")
result = analyze_rules(df, 
                      user_level_badrate_col='USER_LEVEL_BADRATE', 
                      user_target_col='USER_TARGET',
                      hit_date_col='HIT_DATE')

# 3. 查看分析结果
print("\n3. 分析结果概览:")
print(f"共分析 {len(result)} 条规则")
print(f"结果字段: {list(result.columns)}")

# 4. 按实际lift值排序
print("\n4. 按实际lift值排序（前10条）:")
result_sorted_by_actual_lift = result.sort_values(by='actual_lift', ascending=False)
print(result_sorted_by_actual_lift[['rule', 'actual_lift', 'actual_badrate', 'actual_recall', 'actual_precision']].head(10))

# 5. 按预估lift值排序
print("\n5. 按预估lift值排序（前10条）:")
result_sorted_by_estimated_lift = result.sort_values(by='estimated_lift_pred', ascending=False)
print(result_sorted_by_estimated_lift[['rule', 'estimated_lift_pred', 'estimated_badrate_pred', 'estimated_recall_pred', 'estimated_precision_pred']].head(10))

# 6. 对比预估与实际指标
print("\n6. 预估与实际指标对比（前5条）:")
comparison_cols = ['rule', 'estimated_lift_pred', 'actual_lift', 'estimated_badrate_pred', 'actual_badrate']
print(result[comparison_cols].head(5))

# 7. 保存结果到文件
print("\n7. 保存结果到文件...")
output_file = './rule_analysis_result_new.csv'
try:
    result.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"结果已保存到 {output_file}")
except PermissionError:
    print(f"无法写入文件 {output_file}，可能是因为文件已被其他程序打开或没有写入权限")
    print("跳过文件保存步骤")

# 8. 查看规则命中情况
print("\n8. 规则命中情况:")
rule_hit_counts = df['RULE'].value_counts()
print(f"规则命中次数统计（前10条）:")
print(rule_hit_counts.head(10))

# 9. 打印两两规则的相关性矩阵
print("\n9. 两两规则的相关性矩阵:")
correlation_matrix, _ = analyze_rule_correlation(df)
print(correlation_matrix)

# 10. 查看包含命中率及稳定性数据的完整结果
print("\n10. 包含命中率及稳定性数据的完整结果:")
hit_rate_cols = ['rule', 'base_hit_rate', 'current_hit_rate', 'hit_rate_cv', 'max_correlation_value', 'f1']
print(result[hit_rate_cols].head())
