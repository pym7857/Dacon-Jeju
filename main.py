# .venv\Scripts\activate

import gc
import os
import datetime

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import lightgbm as lgb

#! display.max_seq_items 옵션을 None으로 설정하면 생략없이 무제한으로 출력됩니다.
pd.set_option("display.max_seq_items", None)

#! csv to parquet
def csv_to_parquet(csv_path, save_name):
    """
    메모리에 효율적인 데이터 유형을 사용하여 용량을 크게 줄이고 빠른 작업이 가능합니다.
    """
    df = pd.read_csv(csv_path)
    df.to_parquet(f"./{save_name}.parquet")
    del df
    gc.collect()
    print(save_name, "Done.")


# csv_to_parquet(os.path.join("data", "csv", "train.csv"), "train")
# csv_to_parquet(os.path.join("data", "csv", "test.csv"), "test")

#! read parquet
# train = pd.read_parquet(os.path.join("data", "parquet", "train.parquet"))
# test = pd.read_parquet(os.path.join("data", "parquet", "test.parquet"))
# print(train.columns)

data_info = pd.read_csv(os.path.join("data", "csv", "data_info.csv"))
# print(data_info)


def process_dt_and_make_train_pickle(train):
    train["base_date"] = train["base_date"].apply(str)
    # print(train.dtypes)

    # train["base_date"] = pd.to_datetime(train["base_date"])
    # print(train["base_date"].head())

    #! 연,월,일로 나누기
    df_tmp = pd.DataFrame(train.copy())
    df_tmp["datetime"] = [
        datetime.datetime.strptime(timestamp, "%Y%m%d") for timestamp in df_tmp["base_date"]
    ]
    df_tmp["date"] = df_tmp["datetime"].dt.date
    df_tmp["year"] = df_tmp["datetime"].dt.year
    df_tmp["quarter"] = df_tmp["datetime"].dt.quarter
    df_tmp["month"] = df_tmp["datetime"].dt.month
    df_tmp["weekday"] = df_tmp["datetime"].dt.weekday
    df_tmp["day"] = df_tmp["datetime"].dt.day
    # print(df_tmp.head())
    df_tmp.to_pickle("./train.pickle")


# process_dt_and_make_train_pickle(train)

train = pd.read_pickle(os.path.join(""))


# str_col = ["day_of_week", "start_turn_restricted", "end_turn_restricted"]
# for i in str_col:
#     le = LabelEncoder()
#     le = le.fit(train[i])
#     train[i] = le.transform(train[i])

#     for label in np.unique(test[i]):
#         if label not in le.classes_:
#             le.classes_ = np.append(le.classes_, label)
#     test[i] = le.transform(test[i])

# y_train = train["target"]
# X_train = train.drop(
#     [
#         "id",
#         "base_date",
#         "target",
#         "road_name",
#         "start_node_name",
#         "end_node_name",
#         "vehicle_restricted",
#     ],
#     axis=1,
# )
# test = test.drop(
#     ["id", "base_date", "road_name", "start_node_name", "end_node_name", "vehicle_restricted"],
#     axis=1,
# )
# print(X_train.shape)
# print(y_train.shape)
# print(test.shape)

# LR = lgb.LGBMRegressor(random_state=42).fit(X_train, y_train)
# pred = LR.predict(test)
# print(pred)

# sample_submission = pd.read_csv("./sample_submission.csv")
# sample_submission["target"] = pred
# sample_submission.to_csv("./submit.csv", index=False)
# print(sample_submission)
