import pandas as pd

data = pd.read_csv("eval_result.csv", encoding='cp1252')


####################Top 1##################################3
k =400
data.sort_values(['Similarity'],ascending=[0],inplace=True)
grouped = data.groupby(['Tag_1']).head(k)
test_df = pd.DataFrame(columns=['id_1', 'Question', 'id_2','Related_Question', 'Similarity', 'Tag_1', 'Tag_2'])
grouped.to_csv("./top_"+str(k)+"/top_"+str(k)+".csv",index=False)

total_line_num = len(data.groupby(['Tag_1']).head(1))
print ("The total line number is: ", total_line_num)

for index, row in grouped.iterrows():
    if row['Tag_1'] == row['Tag_2']:
        test_df2 = pd.DataFrame({"id_1":[row['id_1']], "Question":[row['Question']], "id_2":[row['id_2']], "Related_Question":[row['Related_Question']], "Similarity":[row['Similarity']], "Tag_1":[row['Tag_1']], "Tag_2":[row['Tag_2']]}) 
        test_df= pd.concat([test_df,test_df2])
        
test_df_3 = test_df.drop_duplicates(subset=["Tag_1","Tag_2"], keep='first', inplace=False)
count = len(test_df_3)

final_result = count / total_line_num
test_df_3.to_csv("./top_"+str(k)+"/group_top_"+str(k)+".csv",index=False)
print ("The precision of the "+str(k)+" is: ", final_result)


#total_line_num = len(data.groupby(['Tag_1']).head(1))
#print ("The total line number is: ", total_line_num)

# =============================================================================
# count = 0
# total_line_num = len(grouped)
# for index, row in grouped.iterrows():
#     if row['Tag_1'] == row['Tag_2']:
#         count = count + 1
# =============================================================================

# =============================================================================
# grouped.duplicated(subset =["Tag_1","Tag_2"], 
#                      keep = 'first') 
# =============================================================================

#grouped_2 = grouped[grouped.duplicated(['Tag_1', 'Tag_2'])]


# =============================================================================
# count = 0
# for index, row in grouped.iterrows():
#     if row['Tag_1'] == row['Tag_2']:
#         count = count + 1
# #count = len(grouped_2)
# =============================================================================

# =============================================================================
# final_result = count / total_line_num
# grouped_2.to_csv("top_"+str(k)+".csv",index=False)
# print ("The precision is: ", final_result)
# # =============================================================================
# =============================================================================
# 
# if 1 > 3.0385314e-05:
#     print ("1")
# else:
#     print ("0")
# =============================================================================
