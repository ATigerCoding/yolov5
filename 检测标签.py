import pandas as pd
import os
import shutil
import cv2


def get_data(path1, path2):
    """
        遍历文件夹下所有文件,并将每个文件的数据以datafrma形式加载合并
    :param path: 测试集真实标签的数据路径
    :return: 测试集预测标签的路径
    """
    data = []
    for file in os.listdir(path1):
        if file.endswith(".txt"):
            file_path = os.path.join(path1, file)
            df = pd.read_csv(file_path, header=None, sep=' ', names=['label', 'x', 'y', 'w', 'h'])
            # 取出label值为 1 3 4 的数据合并
            df = df[(df['label'] == 1) | (df['label'] == 3) | (df['label'] == 5)]
            df['file_name'] = file.split('.')[0]
            data.append(df)
    df_real = pd.concat(data, ignore_index=True)
    df_real = df_real[['file_name', 'label']]
    df_real.groupby('file_name').agg({'label': lambda x: str(set(x))}).reset_index()

    df_pred = pd.read_csv(path2,sep=',')
    df_pred = df_pred[['picture_name', 'type_']]
    df_pred['picture_name'] = df_pred['picture_name'].str.replace('.txt', '')
    df_pred.rename(columns={'picture_name': 'file_name'}, inplace=True)
    df_pred['type_'] = df_pred['type_'].map({'detect_xk': 1, 'detect_fhxk': 3, 'detect_dj': 5})
    df_pred.groupby('file_name').agg({'type_': lambda x: str(set(x))}).reset_index()

    # 对两份数据进行join
    df_merge = pd.merge(df_real, df_pred, on='file_name', how='left')
    df_merge.to_csv(os.path.join(os.path.dirname(path2), 'test_merge.csv'), index=False)


if __name__ == '__main__':
    get_data(path1=r"F:\PythonProject\Job\data_a_b_cut\Dataset-原始\val\labels", path2=r"F:\PythonProject\Job\yolov5-master\runs\detect\exp33\labels2\result.csv")



