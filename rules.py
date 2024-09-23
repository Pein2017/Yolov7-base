import os
from collections import defaultdict

def group_files_by_work_order(dir_path):
    # 存储每个工单号对应的文件路径
    work_order_files = defaultdict(list)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(dir_path):
        if filename.endswith('.txt'):
            # 提取工单号（假设工单号与文件名的开头部分匹配）
            try:
                work_order_number = filename.split('-')[0]
                file_path = os.path.join(dir_path, filename)
                work_order_files[work_order_number].append(file_path)
            except IndexError:
                # 如果文件名格式不符合规范，忽略
                continue
    # 将所有工单号对应的文件路径列表放在一个大列表中
    result = list(work_order_files.values())

    return result

def yolo_result(label_path_lst):
    '''
    image-list[
            box-list[
                yolo result: cls,x,y,w,h,prob
            ]
        ]
    '''
    bbu_sheild_detect_lst = []
    for file_path in label_path_lst:
        image_lst = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(' ')
                image_lst.append(list(map(float, parts)))
        bbu_sheild_detect_lst.append(image_lst)
    return bbu_sheild_detect_lst


def inference(bbu_sheild_detect_lst,threshold):
    '''
    image-list[
            box-list[
                yolo result: cls,x,y,w,h,prob
            ]
        ]
    '''
    # 遍历每一个图片
    max_bbu,max_sheild = 0,0
    bbu_with_shield = False
    special_shot = False
    broad_shot = False
    has_huawei_bbu = False

    if len(bbu_sheild_detect_lst)!=1:
        for image_lst in bbu_sheild_detect_lst:
            all_bbus = 0
            all_shields = 0
            for box_lst in image_lst:
                cls, cx, cy, w, h, c = box_lst
                if c < threshold: continue
                cls = int(cls)

                
                if cls in {0, 3}:  # BBU类
                    all_bbus += 1
                    if cls == 0: has_huawei_bbu = True
                elif cls in {2, 4}:  # Shield类
                    all_shields += 1
            if all_bbus + all_shields >=2:
                broad_shot = True
            if all_bbus + all_shields <=2:
                special_shot = True
                
            

        max_bbu = max(max_bbu,all_bbus)
        max_sheild = max(max_sheild,all_shields)

    bbu_with_shield = max_bbu <= max_sheild and max_sheild!=0  # BBU 是否被 Shield 覆盖
    return bbu_with_shield, special_shot, broad_shot, has_huawei_bbu

def check(files_lst,threshold):
    '''
    yolo返回的分类class对应的名称，txt中存储格式为cls,cx,cy,w,h

    0 huawei_bbu
    1 alx_bbu
    2 huawei_shield
    3 zhongxing_bbu
    4 zhongxing_shield
    '''

    # 判断一个工单内是否有照片为bbu与sheild间隔的，若有则判断正，否则判负
    total_work_orders = len(files_lst)
    negative_work_orders = 0
    pos_file_path_dic = {}
    neg_file_path_dic = {}

    # 遍历每一个工单
    for label_path_lst in files_lst:
        # 遍历每一个图片
        bbu_sheild_detect_lst = yolo_result(label_path_lst)
        bbu_with_shield, special_shot, broad_shot, has_huawei_bbu =inference(bbu_sheild_detect_lst,threshold)
        bbu_with_shield &= special_shot and broad_shot
            
        file_id = label_path_lst[0].split('/')[-1].split('-')[0]
        
        if not bbu_with_shield:
            negative_work_orders += 1
            neg_file_path_dic[file_id] = label_path_lst
        else:
            pos_file_path_dic[file_id] = label_path_lst

    return negative_work_orders, total_work_orders,pos_file_path_dic,neg_file_path_dic

def predict(image_list,threshold=0.41):
    '''
    挡风板bbu预测入口，默认阈值设置为0.41
    Input:
        image_list : 图片list, 由PIL.Image.open函数读取
    Output:
        result:    [['挡风板是否缺失',   '该子问题需要判断', boolean],
                    ['是否有特写照', '该子问题需要判断', boolean],
                    ['是否有全景照', '该子问题需要判断', boolean],
                    ['华为挡风板安装方向是否正确', '该子问题需要判断', boolean],
                    ['是否拧紧螺丝', '该子问题需要判断', boolean]],

        final_bool: boolean
    '''
    # 1. yolo预测输出结果
    # 2. 含有华为bbu的使用箭头
    # 3. 螺丝拧紧预测
    final_bool = False
    
    bbu_with_shield, special_shot, broad_shot, has_huawei_bbu = inference(image_list,threshold)
    
    arrow_direction = True
    if has_huawei_bbu:
        # 判断是否有向上箭头
        arrow_direction = True
        pass
    # 螺丝拧紧预测
    screw_tightness = True

    result = [['挡风板是否缺失','该子问题需要判断',bbu_with_shield],
              ['是否有特写照', '该子问题需要判断', special_shot],
                ['是否有全景照', '该子问题需要判断', broad_shot],
                ['华为挡风板安装方向是否正确', f'该子问题{'不' if not has_huawei_bbu else ""}需要判断', arrow_direction],
                ['是否拧紧螺丝', '该子问题需要判断', screw_tightness]]
    
    final_bool = bbu_with_shield and special_shot and broad_shot and arrow_direction and screw_tightness

    return result, final_bool

if __name__ == '__main__':
    root_path = './runs/detect/v7-p5-lr_1e-3'
    # root_path = './runs/detect/v7-e6e-p6-bs96-ep1000-lr_1e-3'
    # dir_path = root_path+'/pos/labels'

    # 整理数据以一个工单为一组图片，文件命名规则为'工单号-图片序号.txt'，同一工单下会有多张图片
    neg_files_lst = group_files_by_work_order(root_path+'/neg/labels')
    # pos_files_lst = group_files_by_work_order(root_path+'/pos/labels')
    pos_files_lst = group_files_by_work_order(root_path+'/pos/labels')[:len(neg_files_lst)]


    # image_path = '/1004034-0.txt'
    # file_path = dir_path+image_path

    threshold = 0.41

    false_neg, total_pos, _, neg_path_dic_from_pos = check(pos_files_lst,threshold)
    true_neg, total_neg,pos_path_dic_from_neg, _ = check(neg_files_lst,threshold)

    true_pos = total_pos - false_neg
    false_pos = total_neg - true_neg

    print('AI判正的数据中误判比例',false_pos/(true_pos+false_pos))
    print(f"TP:{true_pos},FP:{false_pos}")
    print('正样本中判正比例',true_pos/total_pos)
    print('负样本中判负比例',true_neg/total_neg)
    print("match-ratio:",0.5*(true_pos/total_pos + true_neg/total_neg))