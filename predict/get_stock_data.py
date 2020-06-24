## this source file belong to this project https://github.com/wangshub/RL-Stock
import baostock as bs
import pandas as pd
import os


OUTPUT = './stockdata'


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class Downloader(object):
    def __init__(self,
                 output_dir,
                 date_start='2010-01-01',
                 date_end='2020-03-18'):
        self._bs = bs
        bs.login()
        self.date_start = date_start
        # self.date_end = datetime.datetime.now().strftime("%Y-%m-%d")
        self.date_end = date_end
        self.output_dir = output_dir
        self.fields = "date,code,open,high,low,close,volume,amount," \
                      "adjustflag,turn,tradestatus,pctChg,peTTM," \
                      "pbMRQ,psTTM,pcfNcfTTM,isST"

    def exit(self):
        bs.logout()

    def get_codes_by_date(self, date):
        print(date)
        stock_rs = bs.query_all_stock(date)
        stock_df = stock_rs.get_data()
        print(stock_df)
        return stock_df

    def run(self):
        stock_df = self.get_codes_by_date(self.date_end)
        for index, row in stock_df.iterrows():
            #if ("ST" in row["code_name"]) :
            #    print(f'processing {row["code"]} {row["code_name"]}' + ' ignore it')
            #    continue
            print(f'processing {row["code"]} {row["code_name"]}')
            df_code = bs.query_history_k_data_plus(row["code"], self.fields,
                                                   start_date=self.date_start,
                                                   end_date=self.date_end).get_data()
            df_code.to_csv(f'{self.output_dir}/{row["code"]}.{row["code_name"].replace("*ST","ST")}.csv', index=False)
        self.exit()


if __name__ == '__main__':
    path = './data_sets'
    mkdir(path + '/train/')
    downloader = Downloader(path + '/train', date_start='2019-01-01', date_end='2020-04-20')
    downloader.run()

    mkdir(path + '/test/')
    downloader = Downloader(path + '/test', date_start='2020-04-01', date_end='2020-05-18')
    downloader.run()

