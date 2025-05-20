from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import pyarrow.parquet as pq
import pandas as pd
import json
import time
import pickle

category_mapping = {
    # 电子产品
    "智能手机": "电子产品",
    "笔记本电脑": "电子产品",
    "平板电脑": "电子产品",
    "智能手表": "电子产品",
    "耳机": "电子产品",
    "音响": "电子产品",
    "相机": "电子产品",
    "摄像机": "电子产品",
    "游戏机": "电子产品",
    
    # 服装
    "上衣": "服装",
    "裤子": "服装",
    "裙子": "服装",
    "内衣": "服装",
    "鞋子": "服装",
    "帽子": "服装",
    "手套": "服装",
    "围巾": "服装",
    "外套": "服装",
    
    # 食品
    "零食": "食品",
    "饮料": "食品",
    "调味品": "食品",
    "米面": "食品",
    "水产": "食品",
    "肉类": "食品",
    "蛋奶": "食品",
    "水果": "食品",
    "蔬菜": "食品",
    
    # 家居
    "家具": "家居",
    "床上用品": "家居",
    "厨具": "家居",
    "卫浴用品": "家居",
    
    # 办公
    "文具": "办公",
    "办公用品": "办公",
    
    # 运动户外
    "健身器材": "运动户外",
    "户外装备": "运动户外",
    
    # 玩具
    "玩具": "玩具",
    "模型": "玩具",
    "益智玩具": "玩具",
    
    # 母婴
    "婴儿用品": "母婴",
    "儿童课外读物": "母婴",
    
    # 汽车用品
    "车载电子": "汽车用品",
    "汽车装饰": "汽车用品"
}

def LoadDictMapping(product_dict):
    dict_path = "/home/zouzq/datasets/10G_data_new/product_catalog.json"
    with open(dict_path, encoding="utf-8") as f:
        product_dict = json.load(f)
    return product_dict

# 识别高价格商品
def IdentifyHighPrice(high_price_dict):
    high_price_dict.append(100)
    for i in range(1, 10001):
        price = product_dict["products"][i - 1]["price"]
        if int(price) > 5000:
            high_price_dict.append(1)
        else:
            high_price_dict.append(0)
    print(len(high_price_dict))
    return high_price_dict
    
# 填充频繁项集的集合列表
def TransformIntoSet(p_dict):
    status = p_dict["payment_status"]
    categories = set()
    for item in p_dict["items"]:
        categories.add(category_mapping[product_dict["products"][int(item["id"]) - 1]["category"]])
    processed_data.append(categories)
    
    if status == "已退款":
        part_refund.append(categories)
    elif status == "部分退款":
        full_refund.append(categories)

# 计算支付方式相关，填充字典
def CalculatePayment(p_dict):
    method = p_dict["payment_method"]
    year = int(p_dict["purchase_date"].split('-')[0])
    second = p_dict["purchase_date"].split('-')[1]
    month = 0
    if second == "01" or second == "02" or second == "03":
        month = 1
    elif second == "04" or second == "05" or second == "06":
        month = 2
    elif second == "07" or second == "08" or second == "09":
        month = 3
    elif second == "10" or second == "11" or second == "12":
        month = 4

    for item in p_dict["items"]:
        id = int(item["id"])
        cate = category_mapping[product_dict["products"][id - 1]["category"]]
        if cate not in payment_dict:
            payment_dict[cate] = {}
        
        if method not in payment_dict[cate]:
            payment_dict[cate][method] = 1
        else:
            payment_dict[cate][method] += 1
        
        if high_price_dict[id] == 1:
            if method not in high_price_payment:
                high_price_payment[method] = 1
            else:
                high_price_payment[method] += 1

        if year not in time_dict:
            time_dict[year] = {}
        if month not in time_dict[year]:
            time_dict[year][month] = {}
        if cate not in time_dict[year][month]:
            time_dict[year][month][cate] = 1
        else:
            time_dict[year][month][cate] += 1
    

if __name__ == "__main__":

    file_paths = ["/home/zouzq/datasets/30G_data_new/part-00000.parquet",
                  "/home/zouzq/datasets/30G_data_new/part-00001.parquet",
                  "/home/zouzq/datasets/30G_data_new/part-00002.parquet",
                  "/home/zouzq/datasets/30G_data_new/part-00003.parquet",
                  "/home/zouzq/datasets/30G_data_new/part-00004.parquet",
                  "/home/zouzq/datasets/30G_data_new/part-00005.parquet",
                  "/home/zouzq/datasets/30G_data_new/part-00006.parquet",
                  "/home/zouzq/datasets/30G_data_new/part-00007.parquet",
                  "/home/zouzq/datasets/30G_data_new/part-00008.parquet",
                  "/home/zouzq/datasets/30G_data_new/part-00009.parquet",
                  "/home/zouzq/datasets/30G_data_new/part-00010.parquet",
                  "/home/zouzq/datasets/30G_data_new/part-00011.parquet",
                  "/home/zouzq/datasets/30G_data_new/part-00012.parquet",
                  "/home/zouzq/datasets/30G_data_new/part-00013.parquet",
                  "/home/zouzq/datasets/30G_data_new/part-00014.parquet",
                  "/home/zouzq/datasets/30G_data_new/part-00015.parquet"]
    
    product_dict = {}

    processed_data = []
    product_dict = LoadDictMapping(product_dict)

    high_price_dict = []
    high_price_dict = IdentifyHighPrice(high_price_dict)
    high_price_payment = {}

    payment_dict = {}

    time_dict = {}

    part_refund = []
    full_refund = []
    
    # 填充字典，下一步分析时使用
    for file_path in file_paths:
        print(file_path)
        parqurt_file = pq.ParquetFile(file_path)
        length = parqurt_file.metadata.num_rows

        count = 0
        start_time = time.perf_counter()
        for batch in parqurt_file.iter_batches(batch_size=1000, columns=["purchase_history"]):

            df = batch.to_pandas()

            for _, row in df.iterrows():
                count += 1
                fixed_str = row.to_dict()["purchase_history"].replace('""', '"')
                purchase_dict = json.loads(fixed_str)
                TransformIntoSet(purchase_dict)
                CalculatePayment(purchase_dict)

            if count % 10000 == 0:
                print(str(count) + " Rows has been processed, total " + str(length))
        break

    # 保存文件到指定路径
    with open("processed_data.pkl", "wb") as f:
        pickle.dump(processed_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("processed_data has been dumped into file")

    with open("high_price_payment.pkl", "wb") as f:
        pickle.dump(high_price_payment, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("high_price_payment has been dumped into file")

    with open("payment_dict.pkl", "wb") as f:
        pickle.dump(payment_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("payment_dict has been dumped into file")

    with open("time_dict.pkl", "wb") as f:
        pickle.dump(time_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("time_dict has been dumped into file")

    with open("part_refund.pkl", "wb") as f:
        pickle.dump(part_refund, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("part_refund has been dumped into file")

    with open("full_refund.pkl", "wb") as f:
        pickle.dump(full_refund, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("full_refund has been dumped into file") 
