import numpy as np
import math, pickle

from code.DataLoader import create_output

loader_obj = None
quantity = {}
inv_cost = {}
profit_predict = []
ranking_profit = []
lambda_cons = -1
quantity_value = 200


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


def calculate_discount(p_i, p_max, c_i):
    global lambda_cons
    adoption_prob = lambda_cons * (1.5 - sigmoid(4 * p_i / p_max))

    discount = (- p_i + c_i * math.log(adoption_prob)) / (p_i * math.log(adoption_prob))
    return discount


def calculate_profit():
    global loader_obj, quantity, inv_cost, profit_predict, lambda_cons
    # p_i = 10  # product price
    # p_max = 50 # category max price
    # c_i = 100  # inventory quantity
    # d_i = 0.4  # discount

    # for each user
    for i in range(len(loader_obj.ratings_predict)):
        profit = []
        for j in range(len(loader_obj.ratings_predict[i])):
            p_i = loader_obj.price_dict[j]
            c_i = inv_cost[j]
            for key in loader_obj.new_price_dict:
                obj = loader_obj.new_price_dict[key]
                if j in obj:
                    p_max = loader_obj.max_price[key]
                    break
            d_i = calculate_discount(p_i, p_max, c_i)
            # print("p_i ", p_i)
            # print("p max ", p_max)
            # print("inv cost ", c_i)
            # print("discount ", d_i)
            revenue = p_i * d_i - c_i
            # print("revenue ", revenue)
            rating = loader_obj.ratings_predict[i][j]
            # print("rating ", rating)
            rating_max = 5
            # print("ap....", 1.5 - sigmoid(4 * p_i / p_max))
            # print("ap...  ", lambda_cons * (1.5 - sigmoid(4 * p_i / p_max)))
            adoption_prob = math.pow(lambda_cons * (1.5 - sigmoid(4 * p_i / p_max)), d_i)
            norm_rating = rating / rating_max

            profit_ind = revenue * adoption_prob * norm_rating
            # print("adoption probability ", adoption_prob)
            # print("norm rating ", norm_rating)
            # print("profit ind ", profit_ind)
            profit.append(profit_ind)
        profit_predict.append(profit)
        # print("profit calculated")
        # print(profit_predict)
        print("user ", i)
    return


def calculate_ranking():
    global profit_predict, ranking_profit
    for i in range(len(profit_predict)):
        profit = list(profit_predict[i])
        ranking = np.zeros(len(profit))
        rank = 1
        while rank <= len(profit):
            index = profit.index(max(profit))
            ranking[index] = rank
            rank += 1
            profit[index] = -1
        ranking_profit.append(ranking)
        print("user ranking ", i)
    return


def create_obj():
    global loader_obj
    loader_obj = pickle.load(open("../feature/data_loader.p", "rb"))
    # print(loader_obj.max_price)
    # print(loader_obj.price_dict)

    # for key in x.new_price_dict:
    #     print(key)
    #     print(x.new_price_dict[key])
    #     print(len(x.new_price_dict[key]))
    # print("........")
    count = 0
    for key in loader_obj.ratings_predict:
        # print(key)
        # print(len(key))
        count += 1
        if count == 2:
            break
    count = 0
    for key in loader_obj.ranking:
        # print(key)
        # print(len(key))
        count += 1
        if count == 2:
            break

    return


def create_quantity():
    global loader_obj, quantity, quantity_value
    for i in range(len(loader_obj.price_dict)):
        quantity[i] = quantity_value
    print("quantity initialized")
    print(quantity)
    return


def create_inventory_cost():
    global loader_obj, inv_cost
    for key in loader_obj.price_dict:
        inv_cost[key] = np.random.normal(loader_obj.price_dict[key] / 2, loader_obj.price_dict[key] / 10, 1)[0]
    print("inventory cost initialized")
    print(inv_cost)
    return


def save_file(l_val, q_val):
    global quantity, inv_cost, profit_predict
    all_obj = {}
    all_obj['quantity'] = quantity
    all_obj['inv_cost'] = inv_cost
    all_obj['profit_predict'] = profit_predict
    all_obj['profit_ranking'] = ranking_profit
    pickle.dump(all_obj, open("../feature/profit_feature_" + str(l_val) + "_" + str(q_val) +"_0.5_0.1.p", "wb"))   # name => lambda, quantity, mean, sigma
    return


def create_ds():
    global lambda_cons, loader_obj, quantity, inv_cost, profit_predict, ranking_profit, quantity_value
    lambda_all = [1, 0.7, 0.5, 0.3]
    for val in lambda_all:
        loader_obj = None
        quantity = {}
        inv_cost = {}
        profit_predict = []
        ranking_profit = []
        lambda_cons = -1
        quantity_value = 400

        lambda_cons = val
        create_obj()
        create_quantity()
        create_inventory_cost()
        calculate_profit()
        calculate_ranking()
        save_file(val, quantity_value)


create_ds()