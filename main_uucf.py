import sys
import math
import os
import numpy as np
from sklearn.model_selection import KFold

class Collaborate_Filter:
    def __init__(self):
        self.input_file_name = "data/ratings.tsv"
        self.user_id = "1"
        self.k = 10
        self.dataset = None
        self.uu_dataset = None
        self.ii_dataset = None

    def initialize(self): #gán data
        self.dataset, self.uu_dataset, self.ii_dataset = self.load_data(self.input_file_name)

    def pearson_correlation(self, user1, user2): #tương quan person
        result = 0.0
        user1_data = self.uu_dataset[user1]
        user2_data = self.uu_dataset[user2]

        rx_avg = self.user_average_rating(user1_data)
        ry_avg = self.user_average_rating(user2_data)
        sxy = self.common_items(user1_data, user2_data)

        top_result = 0.0
        bottom_left_result = 0.0
        bottom_right_result = 0.0
        for item in sxy:
            rxs = user1_data[item]
            rys = user2_data[item]
            top_result += (rxs - rx_avg)*(rys - ry_avg)
            bottom_left_result += pow((rxs - rx_avg), 2)
            bottom_right_result += pow((rys - ry_avg), 2)
        bottom_left_result = math.sqrt(bottom_left_result)
        bottom_right_result = math.sqrt(bottom_right_result)
        if top_result != 0:
            result = top_result/(bottom_left_result * bottom_right_result)
        return result

    def user_average_rating(self, user_data): # trung bình đánh giá
        avg_rating = 0.0
        size = len(user_data)
        for (item, rating) in user_data.items():
            avg_rating += float(rating)
        avg_rating /= size * 1.0
        return avg_rating

    def common_items(self, user1_data, user2_data): # get các item chung của 2 user
        result = []
        ht = {}
        for (item, rating) in user1_data.items():
            ht.setdefault(item, 0)
            ht[item] += 1
        for (item, rating) in user2_data.items():
            ht.setdefault(item, 0)
            ht[item] += 1
        for (k, v) in ht.items():
            if v == 2:
                result.append(k)
        return result

    def k_nearest_neighbors(self, user, k): # get k láng giềng gần nhất
        neighbors = []
        result = []

        for (user_id, data) in self.uu_dataset.items():
            if user_id == user:
                continue
            upc = self.pearson_correlation(user, user_id)
            neighbors.append([user_id, upc])
        sorted_neighbors = sorted(neighbors, key=lambda neighbors: (neighbors[1], neighbors[0]), reverse=True)

        for i in range(k):
            if i >= len(sorted_neighbors):
                break
            result.append(sorted_neighbors[i])
        return result

    def list_predicts(self, k_nearest_neighbors, user_id): #get các item được dự đoán
        items_predict = {};
        for (item, data) in self.ii_dataset.items():
            if item not in self.uu_dataset[user_id]:
                items_predict.setdefault(item, self.predict(user_id, item, k_nearest_neighbors))
        return user_id, items_predict



    def predict(self, user, item, k_nearest_neighbors): #tính toán các rating còn thiếu
        valid_neighbors = self.check_neighbors_validattion(item, k_nearest_neighbors)
        if not len(valid_neighbors):
            return 0.0
        top_result = 0.0
        bottom_result = 0.0
        for neighbor in valid_neighbors:
            neighbor_id = neighbor[0]
            neighbor_similarity = neighbor[1]
            rating = self.uu_dataset[neighbor_id][item]
            top_result += neighbor_similarity * rating
            bottom_result += neighbor_similarity
        result = top_result/bottom_result
        return result

    def check_neighbors_validattion(self, item, k_nearest_neighbors): #get list trong k láng giềng đã từng đánh giá item
        result = []
        for neighbor in k_nearest_neighbors:
            neighbor_id = neighbor[0]
            if item in self.uu_dataset[neighbor_id].keys():
                result.append(neighbor)
        return result

    def load_data(self, ratings_file): #xử lý data đầu vào thành các biến
        rating = open(ratings_file, 'r', newline=None)
        dataset = []
        uu_dataset = {}
        ii_dataset = {}
        li = 0
        for line in rating:
            if li != 0:
                row = str(line)
                row = row.split(" ")

                row.pop()
                dataset.append(row)
                uu_dataset.setdefault(row[0], {})
                uu_dataset[row[0]].setdefault(row[1], float(row[2]))
                ii_dataset.setdefault(row[1], {})
                ii_dataset[row[1]].setdefault(row[0], float(row[2]))
            li+=1
        return dataset, uu_dataset, ii_dataset

    def display(self, list_predicts): #hiển thị các item dự đoán
        print('Recommendation for user: ', self.user_id)
        print('List item recommender: ')
        for (item, data) in list_predicts:
            print(item)

    def main(self):
        self.initialize()



if __name__ == '__main__':
    cf = Collaborate_Filter()
    cf.initialize()

    list_user_id = np.arange(1, len(cf.uu_dataset) + 1)
    kfold = KFold(n_splits=5)
    for train_index, test_index in kfold.split(list_user_id):
        train_user_id = list_user_id[train_index]
        test_user_id = list_user_id[test_index]
        for user_id in train_user_id:
            k_nearest_neighbors = cf.k_nearest_neighbors(user_id, 10)
            print(cf.list_predicts(k_nearest_neighbors, user_id))














