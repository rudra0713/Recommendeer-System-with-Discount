import pickle, copy
from code.DataLoader import create_output
import numpy as np


loader_obj = None
quantity = {}
inv_cost = {}
profit_predict = []
ranking_profit = []
REC_ITEM = 10
global_threshold = -1  # 4 from vanilla must be in the profit RS
rec_items_vanilla = []
rec_items_only_profit = []
rec_item_cat_thres = []
rec_item_cat_glob = []
global_trust_kept = 0
product_category = None
sorted_user = None


def within_category_threshold(user_id, product_id, th):
    cat = product_category[product_id]
    cat_len = len(loader_obj.new_price_dict[cat])

    threshold = int(cat_len / th)  # threshold initialized as half of total number of items in a category
    user_ranked = loader_obj.ranking[user_id][product_id]  # get user ranking from vanilla RS
    #print("user id , product id , threshold , user ranked ", user_id, product_id, threshold, user_ranked)
    if user_ranked < threshold:
        return True
    return False


def create_recommendation_cat_threshold(threshold):

    global profit_predict, ranking_profit, quantity, rec_item_cat_thres, sorted_user
    quan_copy = copy.deepcopy(quantity)

    for user in sorted_user:
        top = 1
        rec_prod = []
        profit = 0
        while len(rec_prod) < REC_ITEM and top <= len(ranking_profit[user['user']]):
            top_index = list(ranking_profit[user['user']]).index(top)   # this is the product id
            if within_category_threshold(user['user'], top_index, threshold) and quan_copy[top_index] > 0:
                rec_prod.append(top_index)
                profit += profit_predict[user['user']][top_index]
                quan_copy[top_index] -= 1  # decrement quantity
            top += 1
        rec_item_cat_thres.append(rec_prod)   # not maintaining the serial, still can be found from sorted_user
        print("rec item cat thres ", user['user'])
    return


def create_recommendation_global_threshold():
    global profit_predict, ranking_profit, quantity, loader_obj, rec_items_vanilla, rec_item_cat_glob, global_trust_kept
    quan_copy = copy.deepcopy(quantity)

    for user in sorted_user:
        top = 1
        rec_prod = []
        profit = 0
        rec_van = rec_items_vanilla[user['user']]
        ob_prod_prof = []
        for j in range(len(rec_van)):
            ob = {
                'id': rec_van[j],
                'prof': profit_predict[user['user']][rec_van[j]]
            }
            ob_prod_prof.append(ob)
        sorted_ob_prod_prof = sorted(ob_prod_prof, key=lambda x: x['prof'], reverse=True)

        for j in range(len(sorted_ob_prod_prof)):
            if len(rec_prod) == global_threshold:
                break
            if quan_copy[sorted_ob_prod_prof[j]['id']] > 0:
                rec_prod.append(sorted_ob_prod_prof[j]['id'])
                profit += sorted_ob_prod_prof[j]['prof']
                quan_copy[sorted_ob_prod_prof[j]['id']] -= 1

        if len(rec_prod) == global_threshold:
            global_trust_kept += 1
        # else:
        #     print("failed to keep trust")
        #     print(sorted_ob_prod_prof)
        #     for item in sorted_ob_prod_prof:
        #         print(item['id'], quan_copy[item['id']])
        #     break

        while len(rec_prod) < REC_ITEM and top <= len(ranking_profit[user['user']]):
            product_id = list(ranking_profit[user['user']]).index(top)
            if quan_copy[product_id] > 0 and product_id not in rec_prod:
                rec_prod.append(product_id)
                profit += profit_predict[user['user']][product_id]
                quan_copy[product_id] -= 1
            top += 1
        rec_item_cat_glob.append(rec_prod)

        # if len(rec_prod) != REC_ITEM:
        #     print("DANGER....NOT ENOUGH ITEMS")
        #     break
        print("rect item cat glob ", user['user'])
    return


def create(threshold_val):
    global rec_items_vanilla, rec_item_only_profit, rec_item_cat_glob, rec_item_cat_thres, global_trust_kept, global_threshold
    rec_items_vanilla = []
    rec_item_only_profit = []
    rec_item_cat_thres = []
    rec_item_cat_glob = []
    global_trust_kept = 0
    global_threshold = threshold_val

    return


def save_file(th_cat, th_glb, qu):
    global rec_items_vanilla, rec_item_only_profit, rec_item_cat_glob, rec_item_cat_thres, global_trust_kept
    ob = {
        'trust_category': rec_item_cat_thres,
        'trust_global': rec_item_cat_glob,
        'global_trust_kept': global_trust_kept
    }
    pickle.dump(ob, open("../feature/algo_output_1_sorted_" + str(qu) + "_0.5_0.1_" + str(th_cat) + "_" + str(th_glb) + ".p", "wb"))
    return


def create_algo_output():
    global quantity, inv_cost, profit_predict, ranking_profit, loader_obj, rec_items_vanilla, sorted_user, product_category
    loader_obj = pickle.load(open("../feature/data_loader.p", "rb"))
    product_category = pickle.load(open("../feature/product_category.p", "rb"))

    quantities = [100, 200, 300, 400]
    for qu in quantities:
        ob = pickle.load(open("../feature/profit_feature_1_" + str(qu) + "_0.5_0.1.p", "rb"))
        sorted_user = pickle.load(open("../feature/sorted_user_1_" + str(qu) + "_0.5_0.1_" + ".p", "rb"))

        thresholds_cat = [2, 3, 5]
        thresholds_gl = [4, 6, 8]
        quantity = ob['quantity']
        inv_cost = ob['inv_cost']
        profit_predict = ob['profit_predict']
        ranking_profit = ob['profit_ranking']
        print(quantity)
        for threshold in zip(thresholds_cat, thresholds_gl):
            a,b = threshold  # a => cat, b => global
            algo_o = pickle.load(open("../feature/algo_output_1_" + str(qu) + "_0.5_0.1_" + str(b) + ".p", "rb"))
            create(b)

            rec_items_vanilla = algo_o['vanilla']
            create_recommendation_cat_threshold(a)
            create_recommendation_global_threshold()
            save_file(a, b, qu)

    return


create_algo_output()