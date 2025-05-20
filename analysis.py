from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import pyarrow.parquet as pq
import pandas as pd
import json
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

cate_mapping = {"电子产品":"Electronic Product",
                "服装":"Clothes",
                "食品":"Food",
                "家居":"Furniture",
                "办公":"Office",
                "运动户外":"Outdoor",
                "玩具":"Toy",
                "母婴":"Perinatal Product",
                "汽车用品":"Car Product"}

payment_mapping = {"银联":"Union Pay",
                   "微信支付":"Wechat",
                   "支付宝":"Alipay",
                   "信用卡":"Credit Card",
                   "储蓄卡":"Savings Card",
                   "现金":"Cash",
                   "云闪付":"Quick Pass"}

def fp_growth_analysis(transactions):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
    frequent_itemsets = fpgrowth(df, min_support=0.02, use_colnames=True)
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
    
    final_rules = rules[
        (rules['support'] >= 0.02) &
        (rules['confidence'] >= 0.2) &
        (rules['lift'] > 1)  
    ].sort_values(['support', 'confidence'], ascending=False)
    
    return frequent_itemsets, rules, final_rules
    

if __name__ == "__main__":

    with open("/home/zouzq/datasets/30G_data_new/time_dict.pkl", 'rb') as f: 
        time_dict = pickle.load(f)
    print(time_dict)
    
    for year in time_dict:

        categories = list(time_dict[year][1].keys()) 
        quarters = sorted(time_dict[year].keys())   

        mapped_categories = [cate_mapping.get(cat, cat) for cat in categories] 
        
        quarter_data = {}
        for quarter in quarters:
            quarter_data[quarter] = [time_dict[year][quarter][cat] for cat in categories]
        
        bar_width = 0.2
        index = np.arange(len(categories))
        
        plt.figure(figsize=(12, 6))
        
        for i, quarter in enumerate(quarters):
            plt.bar(index + i * bar_width, 
                    quarter_data[quarter], 
                    width=bar_width,
                    label=f'Q{quarter}')
        
        plt.title(f'{year}', fontsize=14, pad=20)
        plt.xlabel('Categories', fontsize=12)
        plt.ylabel('Times', fontsize=12)
        plt.xticks(index + bar_width * (len(quarters)-1)/2, mapped_categories)
        plt.legend(title='Season')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
    
        
        plt.tight_layout()
        plt.savefig(str(year) + "_purchase.png")

    with open("/home/zouzq/datasets/30G_data_new/high_price_payment.pkl", 'rb') as f: 
        high_price = pickle.load(f)

    mapped_payment = [payment_mapping.get(cat, cat) for cat in high_price] 
    print(mapped_payment)

    print(high_price)

    plt.figure(figsize=(12, 6))

    methods = list(high_price.keys())
    counts = list(high_price.values())

    bars = plt.bar(methods, counts, color="#1fb47b")
    bar_width = 0.2
    index = np.arange(len(methods))

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                height + 10000, 
                f'{height:,}', 
                ha='center', 
                va='bottom',
                fontsize=10)

    plt.title('Statistic', fontsize=15, pad=20)
    plt.xlabel('method', fontsize=12)
    plt.ylabel('amount', fontsize=12)
    plt.xticks(index + bar_width * (len(counts)-1)/2, mapped_payment)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.ylim(min(counts)-100000, max(counts)+100000)

    plt.tight_layout()
    plt.savefig('payment_methods.png', dpi=300, bbox_inches='tight')

    with open("/home/zouzq/datasets/30G_data_new/payment_dict.pkl", 'rb') as f: 
        payment_dict = pickle.load(f)
    print(payment_dict)

    total_transactions = sum(sum(payment_methods.values()) for payment_methods in payment_dict.values())
    print(total_transactions)

    category_totals = {category: sum(payments.values()) for category, payments in payment_dict.items()}

    # 计算每个支付方式的总次数
    payment_totals = {}
    for category in payment_dict:
        for payment in payment_dict[category]:
            if payment not in payment_totals:
                payment_totals[payment] = 0
            payment_totals[payment] += payment_dict[category][payment]

    # 找出支持度 ≥ 0.01 的组合
    high_support_pairs = []
    for category in payment_dict:
        for payment in payment_dict[category]:
            if payment_dict[category][payment] >= 0.01 * total_transactions:
                high_support_pairs.append((category, payment))

    print("支持度 ≥ 0.01 的组合:")
    for pair in high_support_pairs:
        print(pair)

    # 计算 category → payment 的置信度 ≥ 0.6 的规则
    rules_category_to_payment = []
    for category, payment in high_support_pairs:
        confidence = payment_dict[category][payment] / category_totals[category]
        if confidence >= 0.2:
            rules_category_to_payment.append((f"{category} → {payment}", confidence))

    print("\ncategory → payment 置信度 ≥ 0.2 的规则:")
    for rule, conf in rules_category_to_payment:
        print(f"{rule}: {conf:.4f}")

    # 计算 payment → category 的置信度 ≥ 0.6 的规则
    rules_payment_to_category = []
    for category, payment in high_support_pairs:
        confidence = payment_dict[category][payment] / payment_totals[payment]
        if confidence >= 0.2:
            rules_payment_to_category.append((f"{payment} → {category}", confidence))

    print("\npayment → category 置信度 ≥ 0.2 的规则:")
    for rule, conf in rules_payment_to_category:
        print(f"{rule}: {conf:.4f}")

    if not rules_category_to_payment and not rules_payment_to_category:
        print("\n没有满足支持度 ≥ 0.01 且置信度 ≥ 0.2 的关联规则。")


    with open("/home/zouzq/datasets/30G_data_new/full_refund.pkl", 'rb') as f: 
        full_refund = pickle.load(f)
    print("Loaded full refund")

    frequent_itemsets, rules, filtered_rules = fp_growth_analysis(full_refund)
    with open("/home/zouzq/datasets/30G_data_new/1_freq.pkl", 'wb') as f: 
        pickle.dump(frequent_itemsets, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("/home/zouzq/datasets/30G_data_new/1_rules.pkl", 'wb') as f: 
        pickle.dump(rules, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Dumped full refund")



    with open("/home/zouzq/datasets/30G_data_new/part_refund.pkl", 'rb') as f: 
        part_refund = pickle.load(f)
    print("Loaded part refund")

    frequent_itemsets, rules, filtered_rules = fp_growth_analysis(part_refund)
    with open("/home/zouzq/datasets/30G_data_new/2_freq.pkl", 'wb') as f: 
        pickle.dump(frequent_itemsets, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("/home/zouzq/datasets/30G_data_new/2_rules.pkl", 'wb') as f: 
        pickle.dump(rules, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Dumped part refund")


    with open("/home/zouzq/datasets/30G_data_new/processed_data.pkl", 'rb') as f: 
        processed_data = pickle.load(f)
    print("Loaded processed_data")

    frequent_itemsets, rules, filtered_rules = fp_growth_analysis(processed_data)
    with open("/home/zouzq/datasets/30G_data_new/0_freq.pkl", 'wb') as f: 
        pickle.dump(frequent_itemsets, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("/home/zouzq/datasets/30G_data_new/0_rules.pkl", 'wb') as f: 
        pickle.dump(rules, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Dumped processed_data")

    # with open("/home/zouzq/datasets/10G_data_new/2_rules.pkl", 'rb') as f:
    #     rules_2 = pickle.load(f)

    # col_widths = [max(len(str(col)), rules_2[col].astype(str).str.len().max()) for col in rules_2.columns]
    
    # # 打印表头
    # header = "|" + "|".join([str(col).ljust(width) for col, width in zip(rules_2.columns, col_widths)]) + "|"
    # print(header)
    
    # # 打印分隔线（可选）
    # separator = "|" + "|".join(["-" * width for width in col_widths]) + "|"
    # print(separator)
    
    # # 打印数据行
    # for _, row in rules_2.iterrows():
    #     row_str = "|" + "|".join([str(row[col]).ljust(width) for col, width in zip(rules_2.columns, col_widths)]) + "|"
    #     print(row_str)

