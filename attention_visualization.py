
import numpy as np
import pandas as pd
from codecs import open
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



def htmlshow(texts, weights,filename):

    fOut = open(filename, "w", encoding="utf-8")
    part1 ="""
    <html lang="en">
    <head>
    <meta http-equiv="content-type" content="text/html; charset=utf-8">
    <style>
    body {
    font-family: Sans-Serif;
    }
    </style>
    </head>
    <body>
    <h3>
    Heatmaps
    </h3>
    </body>
    <script>
    """
    part2 ="""
    var color = "255,0,0";
    var ngram_length = 3;
    var half_ngram = 1;
    for (var k=0; k < any_text.length; k++) {
    var tokens = any_text[k].split(" ");
    var intensity = new Array(tokens.length);
    var max_intensity = Number.MIN_SAFE_INTEGER;
    var min_intensity = Number.MAX_SAFE_INTEGER;
    for (var i = 0; i < intensity.length; i++) {
    intensity[i] = 0.0;
    for (var j = -half_ngram; j < ngram_length-half_ngram; j++) {
    if (i+j < intensity.length && i+j > -1) {
    intensity[i] += trigram_weights[k][i + j];
    }
    }
    if (i == 0 || i == intensity.length-1) {
    intensity[i] /= 2.0;
    } else {
    intensity[i] /= 3.0;
    }
    if (intensity[i] > max_intensity) {
    max_intensity = intensity[i];
    }
    if (intensity[i] < min_intensity) {
    min_intensity = intensity[i];
    }
    }
    var denominator = max_intensity - min_intensity;
    for (var i = 0; i < intensity.length; i++) {
    intensity[i] = (intensity[i] - min_intensity) / denominator;
    }
    if (k%2 == 0) {
    var heat_text = "<p><br><b>Example:</b><br>";
    } else {
    var heat_text = "<b>Example:</b><br>";
    }
    var space = "";
    for (var i = 0; i < tokens.length; i++) {
    heat_text += "<span style='background-color:rgba(" + color + "," + intensity[i] + ")'>" + space + tokens[i] + "</span>";
    if (space == "") {
    space = " ";
    }
    }
    //heat_text += "<p>";
    document.body.innerHTML += heat_text;
    }
    </script>
    </html>
    """

    putQuote = lambda x: "\"%s\""%x
    textsString = "var any_text = [%s];\n"%(",".join(map(putQuote, texts)))
    weightsString = "var trigram_weights = [%s];\n"%(",".join(map(str,weights)))

    fOut.write(part1)
    fOut.write(textsString)
    fOut.write(weightsString)
    fOut.write(part2)
    fOut.close()


def createselfMAP(id2word,count,total_q,total_c,total_att_A, total_att_M, eval_batch):
    # 随机取一个batch
    batch_id = np.random.choice(list(range(count)))
    print('batch_id:', batch_id)

    # 查询和代码
    batch_query = total_q[batch_id]
    batch_code = total_c[batch_id]
    # 注意力矩阵
    batch_att_M = total_att_M[batch_id]
    # 自注意力矩阵
    batch_att_A = total_att_A[batch_id]

    # 随机取一个样本且保证是正代码,且尽量长
    index = np.random.choice(list(range(eval_batch)))
    print('index:', index)

    # 画图空间不够，去除切割注意力矩阵 0：PAD
    cut_query = [id for id in batch_query[index, :] if id != 0]
    cut_code = [id for id in batch_code[index, :] if id != 0]

    index_dim, column_dim = len(cut_query), len(cut_code)

    query_words = [id2word[id] for id in cut_query]
    print('query_words:', query_words)

    code_tokens = [id2word[id] for id in cut_code]
    print('code_tokens:', code_tokens)

    A= batch_att_A[index, :, :]

    #print(A.shape) 维度

    if A.shape[1]==20:

        #自注意力平均权重
        weights= (np.sum(A,axis=0)/20).tolist()[:index_dim]

        query_texts = [' '.join(query_words)]

        htmlshow(query_texts, [weights], 'qatt.html')

    if A.shape[1]==200:

        # 自注意力平均权重
        weights = (np.sum(A,axis=0)/20).tolist()[:column_dim]

        code_texts = [' '.join(code_tokens)]

        htmlshow(code_texts, [weights], 'catt.html')

    ##############################交互注意力展示##############################
    M = batch_att_M[index, :, :]

    fig = plt.figure(figsize=(12, 12))

    ##############################代码对查询################################
    # 注意力矩阵
    c2q_M = M - M.max(axis=0).reshape(1, 200)
    # (20,200)
    att_c2q_M = np.exp(c2q_M) / (np.exp(c2q_M).sum(axis=0).reshape(1, 200))
    # 切割
    att_c2q_M = att_c2q_M[:index_dim, :column_dim]

    df_c2q = pd.DataFrame(att_c2q_M, index=query_words, columns=code_tokens)

    ax1 = fig.add_subplot(121)

    cax1 = ax1.matshow(df_c2q, interpolation='nearest', cmap='hot_r')
    fig.colorbar(cax1, orientation='horizontal')

    tick_spacing = 1
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    fontdict = {'rotation': 90}
    ax1.set_xticklabels([''] + list(df_c2q.columns), fontdict=fontdict)
    ax1.set_yticklabels([''] + list(df_c2q.index))
    ##############################代码对查询################################

    ##############################查询对代码################################
    # 注意力矩阵
    q2c_M = M - M.max(axis=1).reshape(20, 1)
    # (20,200)
    att_q2c_M = np.exp(q2c_M) / (np.exp(q2c_M).sum(axis=1).reshape(20, 1))

    att_q2c_M = att_q2c_M[: index_dim, :column_dim]

    df_q2c = pd.DataFrame(att_q2c_M, index=query_words, columns=code_tokens)

    ax2 = fig.add_subplot(122)

    cax2 = ax2.matshow(df_q2c, interpolation='nearest', cmap='hot_r')
    fig.colorbar(cax2, orientation='horizontal')

    tick_spacing = 1
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    fontdict = {'rotation': 90}
    ax2.set_xticklabels([''] + list(df_q2c.columns), fontdict=fontdict)
    ax2.set_yticklabels([''] + list(df_q2c.index))
    ##############################查询对代码################################

    plt.savefig('attmap.png')


def createattnMAP(id2word,count,total_q,total_c,total_att_M,eval_batch):

    for i in range(30):
        # 随机取一个batch
        batch_id = np.random.choice(list(range(count)))
        print('batch_id:', batch_id)

        # 查询和代码
        batch_query = total_q[batch_id]
        batch_code = total_c[batch_id]
        # 注意力矩阵
        batch_att_M = total_att_M[batch_id]

        # 随机取一个样本且保证是代码,且尽量长
        index = np.random.choice(list(range(eval_batch)))
        print('index:', index)

        # 画图空间不够，去除切割注意力矩阵 0：PAD
        cut_query = [id for id in batch_query[index,:] if id != 0]
        cut_code  = [id for id in batch_code[index, :] if id != 0]

        index_dim, column_dim = len(cut_query), len(cut_code)

        query_words = [id2word[id] for id in cut_query]
        print('query_words:', query_words)
        code_tokens = [id2word[id] for id in  cut_code]
        print('code_tokens:', code_tokens)

        ##############################交互注意力展示##############################
        M = batch_att_M[index, :, :]

        fig = plt.figure(figsize=(12,12))

        ##############################代码对查询################################
        # 注意力矩阵
        c2q_M = M - M.max(axis=0).reshape(1, 200)
        # (20,200)
        att_c2q_M = np.exp(c2q_M) / (np.exp(c2q_M).sum(axis=0).reshape(1, 200))
        # 切割
        att_c2q_M = att_c2q_M[:index_dim,:column_dim]

        df_c2q = pd.DataFrame(att_c2q_M, index=query_words, columns=code_tokens)

        ax1 = fig.add_subplot(121)

        cax1 = ax1.matshow(df_c2q, interpolation='nearest', cmap='hot_r')
        fig.colorbar(cax1,orientation='horizontal')

        tick_spacing = 1
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

        fontdict = {'rotation': 90}
        ax1.set_xticklabels([''] + list(df_c2q.columns), fontdict=fontdict)
        ax1.set_yticklabels([''] + list(df_c2q.index))
        ##############################代码对查询################################

        ##############################查询对代码################################
        #注意力矩阵
        q2c_M = M - M.max(axis=1).reshape(20, 1)
        # (20,200)
        att_q2c_M = np.exp(q2c_M) / (np.exp(q2c_M).sum(axis=1).reshape(20, 1))

        att_q2c_M = att_q2c_M[:index_dim,:column_dim]

        df_q2c = pd.DataFrame(att_q2c_M, index=query_words, columns=code_tokens)

        ax2 = fig.add_subplot(122)

        cax2 = ax2.matshow(df_q2c, interpolation='nearest', cmap='hot_r')
        fig.colorbar(cax2,orientation='horizontal')

        tick_spacing = 1
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

        fontdict = {'rotation': 90}
        ax2.set_xticklabels([''] + list(df_q2c.columns), fontdict=fontdict)
        ax2.set_yticklabels([''] + list(df_q2c.index))
        ##############################查询对代码################################

        plt.savefig(str(i) + 'attmap.png')
