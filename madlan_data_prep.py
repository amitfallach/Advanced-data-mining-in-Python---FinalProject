import requests
import pandas as pd
import re
import numpy as np
from datetime import datetime, timedelta
import string
import datetime


def prepare_data(data):
    data['publishedDays '] = data['publishedDays '].astype(str).str.extract(r'(\d+\.?\d*)')
    if 'Unnamed: 23' in data.columns:
        data.drop('Unnamed: 23', axis=1, inplace=True)
    data.reset_index(drop=True, inplace=True)
    data = data.dropna(subset=['price']) 
  
# _______________________________________________________________________

    data['City'] = data['City'].str.replace('נהרייה', 'נהריה')
    for col in ['City']:
        if data[col].dtype == 'object':
            data[col] = data[col].str.strip()
# _______________________________________________________________________
    data = data.astype({'room_number':str})
    data['room_number'] = data['room_number'].str.extract(r'([-+]?\d*\.?\d+)')
    data = data.dropna(subset=['price']) 
# _______________________________________________________________________
    def fix_column(df, column_name):
        df = df.astype({column_name:str})
        column_copy = df[column_name].copy()
        column_copy = column_copy.str.extract(r"(\d+)", expand=False)
        column_copy = column_copy.astype('Int64')
        df.loc[:, column_name] = column_copy.copy()

        return df

    data = fix_column(data, 'price')
    data = fix_column(data, 'Area')
# _______________________________________________________________________

    def remove_punctuation(data_name, columns_list):
        cleaned_data = data_name.copy()

        translation_table = str.maketrans('', '', string.punctuation.replace('.', '') + '\n')

        for column in columns_list:
            cleaned_data[column] = cleaned_data[column].astype(str).apply(lambda x: re.sub(r'(?<!\d|\.)[{0}]'.format(string.punctuation.replace('"', '')), ' ', x).translate(translation_table))
        
            cleaned_data[column] = cleaned_data[column].astype(str).apply(lambda x: x.replace('"', ''))

        return cleaned_data

    data = remove_punctuation(data, ['Street', 'city_area','description '])

# _______________________________________________________________________

    def extract_floor_info(value):
        if pd.isnull(value): 
            return 0, 0
        value = str(value)  
        if 'קרקע' in value:
            return 0, 1 
        elif 'קומה' in value:
            match = re.search(r'(\d+)\s*מתוך\s*(\d+)', value)
            if match:
                current_floor = int(match.group(1))
                total_floors = int(match.group(2))
                return current_floor, total_floors
        return None,None 

    data['floor'], data['total_floors'] = zip(*data['floor_out_of'].map(extract_floor_info))
    data['floor'] = pd.to_numeric(data['floor'], errors='coerce').astype('Int64')
    data['total_floors'] = pd.to_numeric(data['total_floors'], errors='coerce').astype('Int64')
    data['floor'] = data['floor'].astype(object).where(data['floor'].notnull(), None)
    data['total_floors'] = data['total_floors'].astype(object).where(data['total_floors'].notnull(), None)
    data = data.drop(['floor_out_of'], axis=1)

# _______________________________________________________________________
    data['entranceDate '] = data['entranceDate '].replace('גמיש', 'flexible')
    data['entranceDate '] = data['entranceDate '].replace('גמיש ', 'flexible')
    data['entranceDate '] = data['entranceDate '].replace('לא צויין', 'not_defined')
    data['entranceDate '] = data['entranceDate '].replace('מיידי', 'Less_than_6 months')

    valid_dates_mask = pd.to_datetime(data['entranceDate '], format='%d/%m/%Y', errors='coerce').notna()
    valid_dates_mask |= pd.to_datetime(data['entranceDate '], format='%Y-%m-%d %H:%M:%S', errors='coerce').notna()

    current_date = datetime.date.today()

    data['time_difference'] = pd.NaT

    valid_dates = pd.to_datetime(data.loc[valid_dates_mask, 'entranceDate '], errors='coerce')
    data.loc[valid_dates_mask, 'time_difference'] = (valid_dates - pd.to_datetime(current_date)).dt.days / 30

    bins = [-float('inf'), 6, 12, float('inf')]
    labels = ['Less_than_6 months', 'months_6_12', 'Above_year']

    data.loc[valid_dates_mask, 'entranceDate '] = pd.cut(data.loc[valid_dates_mask, 'time_difference'], bins=bins, labels=labels)

    data['entranceDate '] = data['entranceDate '].fillna('invalid_value')

    data = data.drop(['time_difference'], axis=1)
# _______________________________________________________________________

    def convert_boolean_to_binary(dataframe, columns, positive_words, negative_words):
        for column in columns:
            for word in positive_words:
                dataframe[column] = dataframe[column].astype(str)
                dataframe[column] = np.where(dataframe[column].str.contains(word, case=False, na=False), 1, dataframe[column])
            for word in negative_words:
                dataframe[column] = dataframe[column].astype(str)
                dataframe[column] = np.where(dataframe[column].str.contains(word, case=False, na=False), 0, dataframe[column])
            dataframe[column] = pd.to_numeric(dataframe[column], errors='coerce').fillna(0).astype(int)
        return dataframe


    positive_words = ['כן', 'יש', 'נגיש', 'TRUE','yes','1','יש מעלית','יש חנייה','יש חניה','יש סורגים','יש מחסן','יש מיזוג אויר','יש מיזוג אוויר','יש מרפסת','יש ממ"ד','נגיש לנכים']
    negative_words = ['לא', 'אין', 'FALSE','no','0','אין מעלית','אין חניה','אין סורגים','אין מחסן','אין מיזוג אויר','אין מרפסת','אין ממ"ד','אין ממ״ד','לא נגיש','לא נגיש לנכים']
    boolean_columns = ['hasElevator ', 'hasParking ', 'hasBars ', 'hasStorage ', 'hasAirCondition ', 'hasBalcony ', 'hasMamad ', 'handicapFriendly ']

    data = convert_boolean_to_binary(data, boolean_columns ,positive_words, negative_words)
# _______________________________________________________________________
    data = data.dropna(subset=['price'])
    data['Area'] = pd.to_numeric(data['Area'], errors='coerce')
    data['room_number'] = pd.to_numeric(data['room_number'], errors='coerce')
    data['number_in_street'] = pd.to_numeric(data['number_in_street'], errors='coerce')
    data['publishedDays '] = pd.to_numeric(data['publishedDays '], errors='coerce')
    col = ['floor','total_floors','price']
    data[col] = data[col].astype(float)
    data = data.dropna(subset = ['condition '])
    data = data[data['price'] <= 15000000]
    data = data.loc[(data['City'] != 'גבעת שמואל') | (data['price'] >= 1000000)]
    data = data.loc[(data['City'] != 'זכרון יעקב') | (data['price'] >= 1000000)]
    data = data.loc[(data['City'] != 'כפר סבא') | (data['price'] >= 1000000)]
    data = data.loc[(data['City'] != 'הרצליה') | (data['price'] >= 1000000)]
    data = data.loc[(data['City'] != 'דימונה') | (data['price'] < 6000000)]
    data = data.loc[(data['City'] != 'ראשון לציון') | (data['price'] < 5000000)] 
    return data
# _______________________________________________________________________





# excel_file = 'Dataset_for_test.xlsx'
#excel_file = 'https://github.com/amitfallach/Advanced-data-mining-in-Python---FinalProject/blob/main/Dataset_for_test.xlsx'
#data = pd.read_excel(excel_file)
#data = pd.read_excel(excel_file, engine='openpyxl')

# excel_file = 'output_all_students_Train_v10.csv'
# data = pd.read_csv(excel_file)

#df = prepare_data(data)