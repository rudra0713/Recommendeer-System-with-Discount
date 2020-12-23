import pickle

import numpy as np, random
import matplotlib.pyplot as plt


def draw_histogram(labels, values, values_2, x_labels, y_labels, file_name, baseline):
    indexes = np.arange(len(labels))

    width = 0.27
    color_arr = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(1, 2 * len(labels) + 1):
        color_arr.append('#' + "%06x" % random.randint(0, 0xFFFFFF)) # generate random colurs for each bar

    # plt.bar(indexes, values, width, color=color_arr[:len(labels)])
    # plt.bar(indexes + width, baseline, width, color=color_arr[len(labels):])

    a = plt.bar(indexes + width, values, width, color='r')
    b = plt.bar(indexes + width * 2, values_2, width, color='g')
    c = plt.bar(indexes, baseline, width, color='b')

    plt.gca().legend(loc='upper left')
    ax.legend((a[0], b[0], c[0]), ('Trust(category)', 'Trust(category_s)', 'Baseline'))
    plt.xticks(indexes + width, labels)
    plt.xlabel(x_labels, fontsize=18)
    plt.ylabel(y_labels, fontsize=18)
    plt.rcParams['figure.figsize'] = (20, 10)
    # plt.title("Quantity vs Average Profit", fontsize=16)
    plt.show()
    fig.savefig("../images/" + file_name, bbox_inches='tight')
# quantity 200, global trust vary from 4, 5, 6, 7


def draw_histogram_2(labels, values, x_labels, y_labels, file_name):
    indexes = np.arange(len(labels))

    width = 0.27
    color_arr = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(1, 2 * len(labels) + 1):
        color_arr.append('#' + "%06x" % random.randint(0, 0xFFFFFF)) # generate random colurs for each bar

    # plt.bar(indexes, values, width, color=color_arr[:len(labels)])
    # plt.bar(indexes + width, baseline, width, color=color_arr[len(labels):])

    a = plt.bar(indexes, values, width, color='r')

    plt.gca().legend(loc='upper left')
    # ax.legend((a[0]), ('Trust kept'))
    plt.xticks(indexes, labels)
    plt.xlabel(x_labels, fontsize=18)
    plt.ylabel(y_labels, fontsize=18)
    plt.rcParams['figure.figsize'] = (20, 10)
    plt.show()
    fig.savefig("../images/" + file_name, bbox_inches='tight')
# quantity 200, global trust vary from 4, 5, 6, 7


def create_graph():
    loader_obj = pickle.load(open("../feature/data_loader.p", "rb"))
    quants = [100, 200, 300, 400]
    profit_trust_global_all = []
    profit_trust_global_sorted_all = []
    trust_all = []
    profit_only_all = []
    profit_trust_cat_all = []
    for quan in quants:
        ds = pickle.load(open("../feature/profit_feature_1_" + str(quan) + "_0.5_0.1.p", "rb"))
        ob = pickle.load(open("../feature/algo_output_1_" + str(quan) + "_0.5_0.1_8.p", "rb"))

        ab = pickle.load(open("../feature/algo_output_1_sorted_" + str(quan) + "_0.5_0.1_5_8.p", "rb"))

        sorted_user = pickle.load(open("../feature/sorted_user_1_" + str(quan) + "_0.5_0.1_.p", "rb"))


        # print(len(ob['trust_global']))
        # print(ob['trust_global'][0])
        # print(ob['profit_only'][0])

        # profit_trust_global = 0
        # profit_only = 0
        # profit_trust_category = 0
        # profit_trust_global_sorted = 0
        # for i in range(len(ob['trust_global'])):
        #     for j, val in enumerate(ob['trust_global'][i]):
        #         profit_trust_global += ds['profit_predict'][i][val]
        # profit_trust_global_all.append(profit_trust_global/10000)
        #
        # for i in range(len(ab['trust_global'])):
        #     for j, val in enumerate(ab['trust_global'][i]):
        #         profit_trust_global_sorted += ds['profit_predict'][sorted_user[i]['user']][val]
        # profit_trust_global_sorted_all.append(profit_trust_global_sorted/10000)
        #
        # for i in range(len(ob['profit_only'])):
        #     for j, val in enumerate(ob['profit_only'][i]):
        #         profit_only += ds['profit_predict'][i][val]
        # profit_only_all.append(profit_only/10000)
        # print("all vals ", quan, profit_trust_global, profit_only)
        trust_all.append(ob['global_trust_kept'] / 10000)

        # print("global trus ", profit_trust_global)
        # print("profit only ", profit_only)
        # print("kept trust for ", ob['global_trust_kept'])
    print(quants)
    # print(profit_trust_global_all)
    #
    # print(profit_trust_cat_all)
    # print(" profit only all ", profit_only_all)
    print("trust kept ", trust_all)

    draw_histogram_2(quants, trust_all, 'Quantity', 'Ratio of trust kept, th = 8', "profit_global_trust_8.pdf")
    # draw_histogram(quants, trust_all, 'Quantity', 'Trust Kept')


def create_graph_cat_trust():
    loader_obj = pickle.load(open("../feature/data_loader.p", "rb"))
    quants = [100, 200, 300, 400]
    profit_trust_global_all = []
    profit_trust_global_sorted_all = []
    trust_all = []
    profit_only_all = []
    profit_trust_cat_all = []
    profit_trust_cat_sorted_all = []

    for quan in quants:
        ds = pickle.load(open("../feature/profit_feature_1_" + str(quan) + "_0.5_0.1.p", "rb"))

        ob = pickle.load(open("../feature/algo_output_1_" + str(quan) + "_0.5_0.1_8.p", "rb"))
        cb = pickle.load(open("../feature/category_ob.p", "rb"))

        ab = pickle.load(open("../feature/algo_output_1_sorted_" + str(quan) + "_0.5_0.1_5_8.p", "rb"))

        sorted_user = pickle.load(open("../feature/sorted_user_1_" + str(quan) + "_0.5_0.1_.p", "rb"))


        # print(len(ob['trust_global']))
        # print(ob['trust_global'][0])
        # print(ob['profit_only'][0])

        profit_trust_global = 0
        profit_only = 0
        profit_trust_category = 0
        profit_trust_global_sorted = 0
        profit_trust_cat_sorted = 0
        key = 'cat_' + str(quan) + "_5"
        for i in range(len(cb[key])):
            for j, val in enumerate(cb[key][i]):
                profit_trust_category += ds['profit_predict'][i][val]
        profit_trust_cat_all.append(profit_trust_category/10000)

        for i in range(len(ab['trust_category'])):
            for j, val in enumerate(ab['trust_category'][i]):
                profit_trust_cat_sorted += ds['profit_predict'][sorted_user[i]['user']][val]
        profit_trust_cat_sorted_all.append(profit_trust_cat_sorted/10000)

        for i in range(len(ob['profit_only'])):
            for j, val in enumerate(ob['profit_only'][i]):
                profit_only += ds['profit_predict'][i][val]
        profit_only_all.append(profit_only/10000)
        # print("all vals ", quan, profit_trust_global, profit_only)
        # trust_all.append(ob['global_trust_kept'] / 10000)

        # print("global trus ", profit_trust_global)
        # print("profit only ", profit_only)
        # print("kept trust for ", ob['global_trust_kept'])
    print(quants)
    print("profit_trust_cat_all ", profit_trust_cat_all)

    print("profit_trust_cat_sorted_all ", profit_trust_cat_sorted_all)
    print("profit_only_all ", profit_only_all)

    draw_histogram(quants, profit_trust_cat_all, profit_trust_cat_sorted_all, 'Quantity', 'Average Proift($), th = 5', "profit_cat_trust_5.pdf", profit_only_all)
    # draw_histogram(quants, trust_all, 'Quantity', 'Trust Kept')


def lambda_draw():
    l_val = [0.3, 0.5, 0.7, 1]
    prof_val = [65.286, 130.724, 248.208, 1615.012]

    draw_histogram_2(l_val, prof_val, 'Lambda value', 'Average Proift($), Quantity = 400', "lambda_400.pdf")
    return


lambda_draw()
# create_graph_cat_trust()
# create_graph()
# draw_histogram([100, 200, 300, 400], [2000, 3000, 4000, 5000],'Quantity', 'Average Profit', [3000, 4000, 5000, 6000])

# for i in range(len(ob['trust_category'])):
#     for j, val in enumerate(ob['trust_category'][i]):
#         profit_trust_category += ds['profit_predict'][i][val]
# profit_trust_cat_all.append(profit_trust_category/10000)


# lambda graphs, quantity = 200 const