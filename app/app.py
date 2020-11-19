from flask import Flask,render_template,request,flash,session,redirect,url_for,send_from_directory,send_file
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

app = Flask(__name__,static_folder="static")
app.config["SECRET_KEY"] = "secret_key"
app.config["UPLOAD_FOLDER"] = "static"
app.config["CLIENT_CSV"] = "Users\riku\Desktop\製作\flask\static"
NAME = ["職種コード","勤務地　市区町村コード",
              "動画ファイル名","勤務地　都道府県コード","会社概要　業界コード","動画タイトル","職種コード","掲載期間　終了日",
              "拠点番号","掲載期間　開始日","公開区分","派遣会社のうれしい特典","お仕事名","（派遣先）職場の雰囲気","動画コメント",
              "休日休暇　備考","（派遣）応募後の流れ","期間・時間　勤務時間","勤務地　備考","（派遣先）配属先部署","お仕事のポイント（仕事PR）",
              "仕事内容","勤務地　最寄駅1（沿線名）","残業なし","オフィスが禁煙・分煙","外資系企業","大手企業","学校・公的機関（官公庁）",
              "Accessのスキルを活かす","Wordのスキルを活かす","Excelのスキルを活かす","PowerPointのスキルを活かす","紹介予定派遣","検索対象エリア",
              "給与/交通費　給与支払区分","就業形態区分","フラグオプション選択","残業月20時間未満","残業月20時間以上","固定残業制","休日休暇(火曜日)",
              "休日休暇(月曜日)","勤務地固定","給与/交通費　給与支払区分","CAD関連のスキルを活かす","勤務先公開","Dip JobsリスティングS","英語力不要",
              "派遣形態","社員食堂あり","派遣スタッフ活躍中","WEB登録OK","制服あり","土日祝休み","DTP関連のスキルを活かす","休日休暇(金曜日)",
              "期間・時間　勤務開始日","休日休暇(祝日)","土日祝のみ勤務",
              "応募資格","勤務地　最寄駅1（駅名）"
              ]
EXTENTENSIONS = "csv"

def file_exist(file):
    file_name = str(file).split("'")[1]
    return file_name

def to_csv(file):
    df = pd.read_csv(file)
    return df

def read_csv():
    df_trainx = pd.read_csv("train_x.csv")
    df_trainy = pd.read_csv("train_y.csv")
    return df_trainx,df_trainy

def concat_df(df_train, df_test):
    df = pd.concat([df_train, df_test])
    df = df.reset_index(drop=True)
    df = df.set_index('お仕事No.')# 列名にidをセット
    return df


def df_drop_na(df):
  for i in df.columns:
    if df[i].isnull().sum()>=924:
      df = df.drop(i,axis=1)
  return df


def split(X,y,test):
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1, test_size=0.3)
    sel = VarianceThreshold(threshold=0.1)
    sel.fit(X_train)
    X_train = sel.transform(X_train)
    X_test = sel.transform(X_test)
    test = sel.transform(test)
    return test

def pred(test):
    test = xgb.DMatrix(test)
    with open('model.pkl', mode='rb') as f: 
        model = pickle.load(f)   
    ans = model.predict(test)   
    return ans       

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/index",methods=["get","post"])
def post():
    if request.method=="POST":
        file = request.files["file_name"]
        if file_exist(file):
            print("ok")
            #fileをcsvに変換
            df_test = to_csv(file)
            df_x, df_y = read_csv()
            #ファイルの結合
            df = concat_df(df_x, df_test)
            #isnullの削除
            df = df_drop_na(df)
            #NAME内の文字の削除
            df = df.drop(NAME, axis=1)
    
            X,test = df[:15853].values,df[15853:].values
            y = df_y["応募数 合計"].values
            test = split(X,y,test)
            print("2")
            ans = pred(test)
            print("pred is ok")
            columns_name = df[15853:].index.values
            df = pd.DataFrame({'お仕事No.':columns_name,
                   '応募数 合計':ans,})
            df.to_csv("static/submission.csv", index=False)
            return render_template(("show.html"))
        else:
            print("error")
            return redirect(url_for("show"))
    else:
        # error_msg = "エラーが発生しました"
        return render_template("index.html")#,name=error_msg)

@app.route("/show")
def show():
    error = "ファイルを選択してください"
    return render_template("error.html",name=error)

@app.route("/download",methods=["get","post"])
def download():
    print("aaaa")
    print(send_from_directory(app.config["CLIENT_CSV"],"submission.csv"))
    return send_from_directory(app.config["UPLOAD_FOLDER"],"submission.csv")


if __name__ == "__main__":
    app.run(debug=True)