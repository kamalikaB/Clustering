#Average_Clustering >> Change Path and Length(4,5,6)

import pandas as pd
import numpy as np
df_0 = pd.read_csv('/home/vickey-vikkrant/Desktop/Heartfailure/heart_failure_clinical_records_dataset.csv')

n=6

#############Preprocessing#########
df_0

df_0 = df_0.drop('DEATH_EVENT', axis=1)
df_0 = df_0.drop('time', axis=1)

# Assuming df_0 is the original DataFrame
df_1 = (df_0 * 10).astype(int)  # Multiply df_0 by 10 and convert to integer

# Perform transformations on specific columns
df_1['platelets'] = (df_1['platelets'] / 10000).astype(int)
df_1['creatinine_phosphokinase'] = (df_1['creatinine_phosphokinase'] / 100).astype(int)
df_1['age'] = (df_1['age'] / 10).astype(int)
df_1['ejection_fraction'] = (df_1['ejection_fraction'] / 10).astype(int)
df_1['serum_sodium'] = (df_1['serum_sodium'] / 100).astype(int)

# Display the resulting DataFrame
df_1


list_of_lists = df_1.values.tolist()  # Converts DataFrame to list of lists
#print(list_of_lists)

def godel_encoding(list_of_lists):
    import sympy

    # Generate enough prime numbers dynamically
    max_length = max(len(ele) for ele in list_of_lists)
    primes = list(sympy.primerange(2, sympy.prime(max_length + 1)))

    godel_numbers = []  # To store results for each list

    for ele in list_of_lists:
        result = 1
        for i, value in enumerate(ele):
            result *= primes[i] ** int(value)
        godel_numbers.append(result)
        #print(result)  # Print each Gödel number

    return godel_numbers  # Return all Gödel numbers

godel_numbers = godel_encoding(list_of_lists)
#print("Gödel Numbers:", godel_numbers)

max_length = max(len(str(num)) for num in godel_numbers)
# Find the next multiple of n for the maximum length
padded_length = ((max_length + n - 1) // n) * n  # Round up to the nearest multiple of n
# Format each Gödel number to the padded length with leading zeros
godel_numbers = [str(num).zfill(padded_length) for num in godel_numbers]



# godel_numbers=df_2['Concatenated'].values.tolist()

#godel_numbers

def split_string(encodings, div):
    r_ind=[]
    enc_length = len(encodings[0])
    # div = math.floor((enc_length)/split)+1
    for i in range(div, enc_length, div):
      r_ind.append(i)
    iclust=[]
    for i in range(len(encodings)):
      s=0
      for j in r_ind:
        iclust.append(encodings[i][s:j])
        s=j
      iclust.append(encodings[i][s:])
      encodings[i]=iclust
      iclust=[]
    return encodings, len(r_ind)+1

import copy
enc = copy.deepcopy(godel_numbers)
split_size = n

# for ele in enc:
#   print(ele)
# print("Encoded:",enc)
split_enc, num_of_splits = split_string(enc,split_size)
#print("Split_encoded",split_enc)

def nullbound(n1,winsize,PS,Rule):
  #  print("PS",PS)
   NS = [0] * n1
   NS[0]=Rule[int(PS[0])*10+int(PS[1])]
   for i in range(1, n1-1):
    NS[i]=Rule[int(PS[i-1])*100+int(PS[i])*10+int(PS[i+1])]
   NS[n1-1]=Rule[int(PS[n1-2])*100+int(PS[n1-1])*10]
  #  print("NS",NS)
   return NS

winsize=3
# def apply_rule(split,rule):
#     final_array = []
#     #print("Split",split)
#     split_list=list(set(split))
#     #split_list.sort(reverse=True)
#     split_len=len(split_list[0])
#     #print("Split list",split_list)
#     # print("Split len",split_len)
#     current_array=[]
#     split_list1=[]
#     for ele in split_list:
#       ele=list(ele)
#       split_list1.append(ele)
#     split_list=split_list1
#     #print("Split list",split_list)

#     while(split_list):
#       curr_element=split_list[0]
#       #curr_element=list(curr_element)
#       #print("Current element",curr_element)
#       flag=0
#       while(not flag):
#         if current_array == []:
#           current_element1=[]
#           for c in curr_element:
#             current_element1.append(int(c))
#           current_array.append(current_element1)

#           #current_array.append(curr_element)
#           #print("Current array",current_array)
#           split_list.remove(curr_element)
#         else:
#           next_element=nullbound(split_len,winsize,curr_element,rule)
#           #print("Next element",next_element)

#           if (next_element not in current_array):
#             #print("Next element no in cycle")
#             #print("Split:",split_list)
#             next_element1 = [str(x) for x in next_element]
#             if next_element1 in split_list:
#                 #print("next element in dataset")
#                 split_list.remove(next_element1)
#                 current_array.append(next_element)
#             #curr_element=next_element
#             import copy

#             # Assuming next_element is a list (or a list of lists)
#             curr_element = copy.deepcopy(next_element)
#                 #print("Current element",curr_element)
#           else:
#               # print("last Current element",curr_element)
#               flag=1
#               #current_array.append(curr_element)
#       final_array.append(current_array)
#       #print("cycle",current_array)
#       current_array=[]
#     return final_array
import copy

def apply_rule(split, rule):
    final_array = []
    split_list = list(set(split))  # Remove duplicates
    split_list.sort()  # Ensure a deterministic order
    split_len = len(split_list[0])  # Assuming all elements have the same length
    current_array = []
    split_list1 = []

    for ele in split_list:
        ele = list(ele)
        split_list1.append(ele)
    split_list = split_list1

    while split_list:
        curr_element = split_list[0]
        flag = 0
        
        while not flag:
            if not current_array:
                # Add current element to the array
                current_element1 = [int(c) for c in curr_element]
                current_array.append(current_element1)

                # Remove from split_list
                split_list.remove(curr_element)
            else:
                # Assuming 'nullbound' function processes and generates next element
                next_element = nullbound(split_len, winsize, curr_element, rule)
                
                if next_element not in current_array:
                    # Ensure the next element is in split_list before adding
                    next_element1 = [str(x) for x in next_element]
                    if next_element1 in split_list:
                        split_list.remove(next_element1)
                        current_array.append(next_element)
                    
                    # Deep copy the next element for the next iteration
                    curr_element = copy.deepcopy(next_element)
                else:
                    flag = 1

        final_array.append(current_array)
        current_array = []

    return final_array

# def Stage1(rule1,rule2,rule3,enc_stage1,split,min_split_size,max_split_size):
#   fc = {}
#   tr = []
#   #print("\ninput to stage 1",enc_stage1)
#   #***********************applying rule to each split********************

#   for i in range(split):
#     for j in range(len(enc_stage1)):
#       tr.append(enc_stage1[j][i])
#     #print("split_data",tr)
#     fc[i]=apply_rule(tr,rule1)
#     tr = []
#   return fc

def Stage1(rule1, rule2, rule3, enc_stage1, split, min_split_size, max_split_size):
    fc = {}
    tr = []

    # Make sure the input data is ordered consistently
    enc_stage1 = sorted(enc_stage1)  # Sort enc_stage1 to ensure consistent order

    # *********************** Applying rule to each split ********************
    for i in range(split):
        # Initialize tr as an empty list for each split iteration
        tr = []
        
        # Collect elements for the current split and append them in a consistent order
        for j in range(len(enc_stage1)):
            tr.append(enc_stage1[j][i])

        # Apply the rule to the collected data for this split
        fc[i] = apply_rule(tr, rule1)

    return fc

def rg(p,d=10):
    left = 1
    right = 1
    m = left + right + 1
    # n = cell_length  # Number of cells

    # Parse the parameters from the input string
    words = p.split(",")
    param = [int(word.strip()) for word in words]

    # Generate the rule from the parameters
    Rule = []
    for x in range(d):
        for y in range(d):
            for z in range(d):
                rule = (param[0] * x * y * z + param[1] * x * y + param[2] * x * z +
                        param[3] * z * y + param[4] * x + param[5] * y + param[6] * z + param[7]) % d
                Rule.append(rule)
    #print("\nRule:",Rule)
    return Rule

def cy_enc(l, index):
    # Determine the number of digits in l
    length = len(str(l))
    # Format the index with leading zeros based on the length
    return f"{index:0{length}d}"

def Stage_part2(R1,R2,R3,dataset2,min_split_size,max_split_size):
  print("Rule 3 Used")
  with open('/home/vickey-vikkrant/Desktop/Heartfailure/HF_Avg_l6.txt', 'a') as file:
        file.write(f"Rule3  used     ")
        file.close()
  #print(dataset2)
  split_size=n
  max_len = max(len(s) for s in dataset2)  # Find the maximum string length
  target_len = ((max_len + n - 1) // n) * n
  dataset2_n=[s.zfill(target_len) for s in dataset2]
  #encodings = [str(e) for e in encodings]
  split_enc2, num_of_splits2 = split_string(dataset2_n,split_size)
  #print(split_enc2)
  Stage2_part2_output1=Stage1(R1,R2,R3,split_enc2,num_of_splits2,min_split_size,max_split_size)
  stage2_output2=[]
  stage2_output2=Stage2(R1,R2,R3,Stage2_part2_output1,split_enc2,min_split_size,max_split_size)
  return stage2_output2

def custom_sort_by_average(cluster):
    def compute_average(lst):
        # Convert elements to integers
        lst = list(map(int, lst))
        # Compute sum and length manually
        total = 0
        for num in lst:
            total += num
        avg = total / len(lst)
        return avg

    # Custom Bubble Sort on Cluster based on Average
    n = len(cluster)
    for i in range(n):
        for j in range(0, n - i - 1):
            avg1 = compute_average(cluster[j])
            avg2 = compute_average(cluster[j + 1])
            if avg1 > avg2 or (avg1 == avg2 and cluster[j] > cluster[j + 1]):
                # Swap if the first is greater or if averages are equal but lexicographically out of order
                cluster[j], cluster[j + 1] = cluster[j + 1], cluster[j]
    return cluster

# Example Usage
# icluster_Avg_sort = [
#     ["3", "4", "5"],  # Average: 4.0
#     ["2", "4", "6"],  # Average: 4.0
#     ["1", "2", "3"],  # Average: 2.0
#     ["4", "4", "4"],  # Average: 4.0
# ]

# sorted_cluster = custom_sort_by_average(icluster_Avg_sort)
# print(sorted_cluster)

def Stage2(R1,R2,R3,cluster1,dataset1,min_split_size,max_split_size):
  #print("input to stage2 dataset",dataset1)

  #***************Sorting the cluster based on median**********************

  for i in range(len(cluster1)):
    #icluster_Med_sort=Median_Cycles(cluster1[i])
    import statistics
    cluster1[i] = [[''.join(map(str, item)) for item in sublist] for sublist in cluster1[i]]
    #icluster_Avg_sort = sorted(cluster1[i], key=lambda sublist: statistics.mean(map(int, sublist)))
    #icluster_Avg_sort = sorted(cluster1[i], key=lambda sublist: (round(statistics.mean(map(int, sublist)), 5), sublist))
    icluster_Avg_sort = custom_sort_by_average(cluster1[i])
    #print(icluster_Avg_sort)

    iclust=copy.deepcopy(icluster_Avg_sort)
    s=len(iclust)
    for j in range(len(dataset1)):
      plt=dataset1[j][i]
      l=0
      while(plt not in iclust[l]):
        l+=1
      dataset1[j][i]= cy_enc(s,l)
  #print(dataset1)
  #************************************merging all split into one****************
  for i in range(len(dataset1)):
    iclust=""
    for j in dataset1[i]:
      iclust+=j
    dataset1[i]=iclust
  enc_data=[]
  init_clusters=[]

  if(len(dataset1[0])<min_split_size):
    dataset1 = [elem[:n].ljust(n, '0') for elem in dataset1]
  #If merged data length less than maximum possible cell size  apply rule 2
  if(len(dataset1[0])<=max_split_size):
    init_clusters=apply_rule(dataset1,R2)
    result = []
    for sublist in init_clusters:
        sub_result = [''.join(map(str, lst)) for lst in sublist]
        result.append(sub_result)
    init_clusters=result.copy()
    import statistics

    enc_data=copy.deepcopy(dataset1)

  else:
    Stage2_output_reduced=Stage_part2(R1,R2,R3,dataset1,min_split_size,max_split_size)
    return Stage2_output_reduced
  Stage2_output=[]
  Stage2_output.append(dataset1)
  Stage2_output.append(init_clusters)
  return Stage2_output

def stage3_Avg(cycles_ex, Dataset_ex,num_clusters):
  #print("Orginal Dataset:",Dataset_ex)
  k = num_clusters
  cycles_ex = [[int(element) for element in sublist] for sublist in cycles_ex]
  #print("Clusters",cycles_ex)
  #print("Dataset",Dataset_ex)
  #cycles_ex.sort(key=len, reverse=True)
  cycles_ex.sort(key=lambda x: (-len(x), x))
  while len(cycles_ex) > k:
      #last_list = cycles_ex[-1]
      last_list = cycles_ex[-1][:]
      while last_list:
          first_element = last_list[0]
          differences = []
          for sublist in cycles_ex[:-1]:
              old_avg = round(sum(sublist) / len(sublist))

              new_avg = round((sum(sublist) + first_element) / (len(sublist) + 1))

              difference = round(abs(new_avg - old_avg))
              differences.append(difference)
          #print(f"Differences for adding {first_element}: {differences}")
          #min_index = differences.index(min(differences))
          min_value = min(differences)
          candidates = [i for i, diff in enumerate(differences) if diff == min_value]
          min_index = min(candidates, key=lambda idx: sum(cycles_ex[idx]))
          cycles_ex[min_index].append(first_element)
          last_list.remove(first_element)
      if not last_list:
          cycles_ex.pop()
      #cycles_ex.sort(key=len, reverse=True)
      cycles_ex.sort(key=lambda x: (-len(x), x))
  #print("\nFinal data (sorted by number of elements):", cycles_ex)
  Dataset_ex = list(map(int, Dataset_ex))
  for i, element in enumerate(Dataset_ex):
      found = False
      for index, sublist in enumerate(cycles_ex):
          if element in sublist:
              Dataset_ex[i] = index
              found = True
              break
      if not found:
          Dataset_ex[i] = -1
  #print("Updated Dataset_ex:", Dataset_ex)
  return Dataset_ex

# stage3_Avg(cycles_ex, Dataset_ex)

from itertools import combinations, permutations
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
import copy
def cellular_automata_clustering(rule_list, split, encoding,n,num_clusters,min_split_size,max_split_size):
  #rules_comb = list(combinations(rule_list, 2))
  #print(rules_comb)
  #rules_comb=[('0,0,0,0,0,1,9,1','0,0,0,0,0,9,6,8')]
  import random
  #R = random.choice(rules_comb)
  enc1 = copy.deepcopy(encoding)
  R=[]
  R.append(random.choice(rule_list))
  #R.append('0,0,0,0,0,1,3,2')
  rule1=rg(R[0])
  R.append(random.choice(rule_list))
  #R.append('0,0,0,0,0,1,3,2')

  rule2=rg(R[1])

  random_rule = random.choice(rule_list)

  #random_rule='0,0,0,0,2,9,0,5'
  rule3=rg(random_rule)

  fc = Stage1(rule1,rule2,rule3,enc1,split,min_split_size,max_split_size)
  # for index, (key, value) in enumerate(fc.items()):
  #   print(f"Split {index}: Key = {key}, Cycles = {value}")

  stage2_output=[]
  stage2_output=Stage2(rule1,rule2,rule3,fc,enc1,min_split_size,max_split_size)
  stage2_dataset=stage2_output[0]
  stage2_cycles=stage2_output[1]
  # print("stage2_dataset",stage2_dataset)
  # print("stage2_cycles",stage2_cycles)

  #num_clusters=3
  enc_data_=stage3_Avg(stage2_cycles,stage2_dataset,num_clusters)
  #print(enc_data_)
  X=df_0.to_numpy()
  # if(silhouette_score(X,enc_data_,metric="euclidean")>0.6):
  #   print("Rule1:",R[0],"Rule2:",R[1],"Rule3:",random_rule)
  #   print("\tsilhouette:CA",silhouette_score(X,enc_data_,metric="euclidean"),"\tdavies:CA",davies_bouldin_score(X,enc_data_),"\tcalinski:CA",calinski_harabasz_score(X,enc_data_))

  try :
    CA_sill_new=silhouette_score(X,enc_data_,metric="euclidean")
    # print("davies:CA",davies_bouldin_score(X,enc_data_))
    # print("calinski:CA",calinski_harabasz_score(X,enc_data_))
    print("Rule1:",R[0],"Rule2:",R[1],"Rule3:",random_rule,"\tsilhouette:CA",silhouette_score(X,enc_data_,metric="euclidean"),"\tdavies:CA",davies_bouldin_score(X,enc_data_),"\tcalinski:CA",calinski_harabasz_score(X,enc_data_))
    # silhouette = silhouette_score(X, enc_data_, metric="euclidean")
    # davies = davies_bouldin_score(X, enc_data_)
    # calinski = calinski_harabasz_score(X, enc_data_)
    with open('/home/vickey-vikkrant/Desktop/Heartfailure/HF_Avg_l6.txt', 'a') as file:
      file.write(f"Rule1: {R[0]}, Rule2: {R[1]}, Rule3: {random_rule}\t\tsilhouette:CA,{silhouette_score(X,enc_data_,metric="euclidean")},\tdavies:CA,{davies_bouldin_score(X,enc_data_)},\tcalinski:CA,{calinski_harabasz_score(X,enc_data_)},\n")
  except:
    CA_sill_new = 0

rule_list=['0,0,0,0,0,1,0,0',
'0,0,0,0,0,1,0,1',
'0,0,0,0,0,1,0,2',
'0,0,0,0,0,1,0,3',
'0,0,0,0,0,1,0,4',
'0,0,0,0,0,1,0,5',
'0,0,0,0,0,1,0,6',
'0,0,0,0,0,1,0,7',
'0,0,0,0,0,1,0,8',
'0,0,0,0,0,1,0,9',
'0,0,0,0,0,1,1,0',
'0,0,0,0,0,1,1,1',
'0,0,0,0,0,1,1,2',
'0,0,0,0,0,1,1,3',
'0,0,0,0,0,1,1,4',
'0,0,0,0,0,1,1,5',
'0,0,0,0,0,1,1,6',
'0,0,0,0,0,1,1,7',
'0,0,0,0,0,1,1,8',
'0,0,0,0,0,1,1,9',
'0,0,0,0,0,1,2,0',
'0,0,0,0,0,1,2,1',
'0,0,0,0,0,1,2,2',
'0,0,0,0,0,1,2,3',
'0,0,0,0,0,1,2,4',
'0,0,0,0,0,1,2,5',
'0,0,0,0,0,1,2,6',
'0,0,0,0,0,1,2,7',
'0,0,0,0,0,1,2,8',
'0,0,0,0,0,1,2,9',
'0,0,0,0,0,1,3,0',
'0,0,0,0,0,1,3,1',
'0,0,0,0,0,1,3,2',
'0,0,0,0,0,1,3,3',
'0,0,0,0,0,1,3,4',
'0,0,0,0,0,1,3,5',
'0,0,0,0,0,1,3,6',
'0,0,0,0,0,1,3,7',
'0,0,0,0,0,1,3,8',
'0,0,0,0,0,1,3,9',
'0,0,0,0,0,1,4,0',
'0,0,0,0,0,1,4,1',
'0,0,0,0,0,1,4,2',
'0,0,0,0,0,1,4,3',
'0,0,0,0,0,1,4,4',
'0,0,0,0,0,1,4,5',
'0,0,0,0,0,1,4,6',
'0,0,0,0,0,1,4,7',
'0,0,0,0,0,1,4,8',
'0,0,0,0,0,1,4,9',
'0,0,0,0,0,1,5,0',
'0,0,0,0,0,1,5,1',
'0,0,0,0,0,1,5,2',
'0,0,0,0,0,1,5,3',
'0,0,0,0,0,1,5,4',
'0,0,0,0,0,1,5,5',
'0,0,0,0,0,1,5,6',
'0,0,0,0,0,1,5,7',
'0,0,0,0,0,1,5,8',
'0,0,0,0,0,1,5,9',
'0,0,0,0,0,1,6,0',
'0,0,0,0,0,1,6,1',
'0,0,0,0,0,1,6,2',
'0,0,0,0,0,1,6,3',
'0,0,0,0,0,1,6,4',
'0,0,0,0,0,1,6,5',
'0,0,0,0,0,1,6,6',
'0,0,0,0,0,1,6,7',
'0,0,0,0,0,1,6,8',
'0,0,0,0,0,1,6,9',
'0,0,0,0,0,1,7,0',
'0,0,0,0,0,1,7,1',
'0,0,0,0,0,1,7,2',
'0,0,0,0,0,1,7,3',
'0,0,0,0,0,1,7,4',
'0,0,0,0,0,1,7,5',
'0,0,0,0,0,1,7,6',
'0,0,0,0,0,1,7,7',
'0,0,0,0,0,1,7,8',
'0,0,0,0,0,1,7,9',
'0,0,0,0,0,1,8,0',
'0,0,0,0,0,1,8,1',
'0,0,0,0,0,1,8,2',
'0,0,0,0,0,1,8,3',
'0,0,0,0,0,1,8,4',
'0,0,0,0,0,1,8,5',
'0,0,0,0,0,1,8,6',
'0,0,0,0,0,1,8,7',
'0,0,0,0,0,1,8,8',
'0,0,0,0,0,1,8,9',
'0,0,0,0,0,1,9,0',
'0,0,0,0,0,1,9,1',
'0,0,0,0,0,1,9,2',
'0,0,0,0,0,1,9,3',
'0,0,0,0,0,1,9,4',
'0,0,0,0,0,1,9,5',
'0,0,0,0,0,1,9,6',
'0,0,0,0,0,1,9,7',
'0,0,0,0,0,1,9,8',
'0,0,0,0,0,1,9,9',
'0,0,0,0,0,3,0,0',
'0,0,0,0,0,3,0,1',
'0,0,0,0,0,3,0,2',
'0,0,0,0,0,3,0,3',
'0,0,0,0,0,3,0,4',
'0,0,0,0,0,3,0,5',
'0,0,0,0,0,3,0,6',
'0,0,0,0,0,3,0,7',
'0,0,0,0,0,3,0,8',
'0,0,0,0,0,3,0,9',
'0,0,0,0,0,3,1,0',
'0,0,0,0,0,3,1,1',
'0,0,0,0,0,3,1,2',
'0,0,0,0,0,3,1,3',
'0,0,0,0,0,3,1,4',
'0,0,0,0,0,3,1,5',
'0,0,0,0,0,3,1,6',
'0,0,0,0,0,3,1,7',
'0,0,0,0,0,3,1,8',
'0,0,0,0,0,3,1,9',
'0,0,0,0,0,3,2,0',
'0,0,0,0,0,3,2,1',
'0,0,0,0,0,3,2,2',
'0,0,0,0,0,3,2,3',
'0,0,0,0,0,3,2,4',
'0,0,0,0,0,3,2,5',
'0,0,0,0,0,3,2,6',
'0,0,0,0,0,3,2,7',
'0,0,0,0,0,3,2,8',
'0,0,0,0,0,3,2,9',
'0,0,0,0,0,3,3,0',
'0,0,0,0,0,3,3,1',
'0,0,0,0,0,3,3,2',
'0,0,0,0,0,3,3,3',
'0,0,0,0,0,3,3,4',
'0,0,0,0,0,3,3,5',
'0,0,0,0,0,3,3,6',
'0,0,0,0,0,3,3,7',
'0,0,0,0,0,3,3,8',
'0,0,0,0,0,3,3,9',
'0,0,0,0,0,3,4,0',
'0,0,0,0,0,3,4,1',
'0,0,0,0,0,3,4,2',
'0,0,0,0,0,3,4,3',
'0,0,0,0,0,3,4,4',
'0,0,0,0,0,3,4,5',
'0,0,0,0,0,3,4,6',
'0,0,0,0,0,3,4,7',
'0,0,0,0,0,3,4,8',
'0,0,0,0,0,3,4,9',
'0,0,0,0,0,3,5,0',
'0,0,0,0,0,3,5,1',
'0,0,0,0,0,3,5,2',
'0,0,0,0,0,3,5,3',
'0,0,0,0,0,3,5,4',
'0,0,0,0,0,3,5,5',
'0,0,0,0,0,3,5,6',
'0,0,0,0,0,3,5,7',
'0,0,0,0,0,3,5,8',
'0,0,0,0,0,3,5,9',
'0,0,0,0,0,3,6,0',
'0,0,0,0,0,3,6,1',
'0,0,0,0,0,3,6,2',
'0,0,0,0,0,3,6,3',
'0,0,0,0,0,3,6,4',
'0,0,0,0,0,3,6,5',
'0,0,0,0,0,3,6,6',
'0,0,0,0,0,3,6,7',
'0,0,0,0,0,3,6,8',
'0,0,0,0,0,3,6,9',
'0,0,0,0,0,3,7,0',
'0,0,0,0,0,3,7,1',
'0,0,0,0,0,3,7,2',
'0,0,0,0,0,3,7,3',
'0,0,0,0,0,3,7,4',
'0,0,0,0,0,3,7,5',
'0,0,0,0,0,3,7,6',
'0,0,0,0,0,3,7,7',
'0,0,0,0,0,3,7,8',
'0,0,0,0,0,3,7,9',
'0,0,0,0,0,3,8,0',
'0,0,0,0,0,3,8,1',
'0,0,0,0,0,3,8,2',
'0,0,0,0,0,3,8,3',
'0,0,0,0,0,3,8,4',
'0,0,0,0,0,3,8,5',
'0,0,0,0,0,3,8,6',
'0,0,0,0,0,3,8,7',
'0,0,0,0,0,3,8,8',
'0,0,0,0,0,3,8,9',
'0,0,0,0,0,3,9,0',
'0,0,0,0,0,3,9,1',
'0,0,0,0,0,3,9,2',
'0,0,0,0,0,3,9,3',
'0,0,0,0,0,3,9,4',
'0,0,0,0,0,3,9,5',
'0,0,0,0,0,3,9,6',
'0,0,0,0,0,3,9,7',
'0,0,0,0,0,3,9,8',
'0,0,0,0,0,3,9,9',
'0,0,0,0,0,7,0,0',
'0,0,0,0,0,7,0,1',
'0,0,0,0,0,7,0,2',
'0,0,0,0,0,7,0,3',
'0,0,0,0,0,7,0,4',
'0,0,0,0,0,7,0,5',
'0,0,0,0,0,7,0,6',
'0,0,0,0,0,7,0,7',
'0,0,0,0,0,7,0,8',
'0,0,0,0,0,7,0,9',
'0,0,0,0,0,7,1,0',
'0,0,0,0,0,7,1,1',
'0,0,0,0,0,7,1,2',
'0,0,0,0,0,7,1,3',
'0,0,0,0,0,7,1,4',
'0,0,0,0,0,7,1,5',
'0,0,0,0,0,7,1,6',
'0,0,0,0,0,7,1,7',
'0,0,0,0,0,7,1,8',
'0,0,0,0,0,7,1,9',
'0,0,0,0,0,7,2,0',
'0,0,0,0,0,7,2,1',
'0,0,0,0,0,7,2,2',
'0,0,0,0,0,7,2,3',
'0,0,0,0,0,7,2,4',
'0,0,0,0,0,7,2,5',
'0,0,0,0,0,7,2,6',
'0,0,0,0,0,7,2,7',
'0,0,0,0,0,7,2,8',
'0,0,0,0,0,7,2,9',
'0,0,0,0,0,7,3,0',
'0,0,0,0,0,7,3,1',
'0,0,0,0,0,7,3,2',
'0,0,0,0,0,7,3,3',
'0,0,0,0,0,7,3,4',
'0,0,0,0,0,7,3,5',
'0,0,0,0,0,7,3,6',
'0,0,0,0,0,7,3,7',
'0,0,0,0,0,7,3,8',
'0,0,0,0,0,7,3,9',
'0,0,0,0,0,7,4,0',
'0,0,0,0,0,7,4,1',
'0,0,0,0,0,7,4,2',
'0,0,0,0,0,7,4,3',
'0,0,0,0,0,7,4,4',
'0,0,0,0,0,7,4,5',
'0,0,0,0,0,7,4,6',
'0,0,0,0,0,7,4,7',
'0,0,0,0,0,7,4,8',
'0,0,0,0,0,7,4,9',
'0,0,0,0,0,7,5,0',
'0,0,0,0,0,7,5,1',
'0,0,0,0,0,7,5,2',
'0,0,0,0,0,7,5,3',
'0,0,0,0,0,7,5,4',
'0,0,0,0,0,7,5,5',
'0,0,0,0,0,7,5,6',
'0,0,0,0,0,7,5,7',
'0,0,0,0,0,7,5,8',
'0,0,0,0,0,7,5,9',
'0,0,0,0,0,7,6,0',
'0,0,0,0,0,7,6,1',
'0,0,0,0,0,7,6,2',
'0,0,0,0,0,7,6,3',
'0,0,0,0,0,7,6,4',
'0,0,0,0,0,7,6,5',
'0,0,0,0,0,7,6,6',
'0,0,0,0,0,7,6,7',
'0,0,0,0,0,7,6,8',
'0,0,0,0,0,7,6,9',
'0,0,0,0,0,7,7,0',
'0,0,0,0,0,7,7,1',
'0,0,0,0,0,7,7,2',
'0,0,0,0,0,7,7,3',
'0,0,0,0,0,7,7,4',
'0,0,0,0,0,7,7,5',
'0,0,0,0,0,7,7,6',
'0,0,0,0,0,7,7,7',
'0,0,0,0,0,7,7,8',
'0,0,0,0,0,7,7,9',
'0,0,0,0,0,7,8,0',
'0,0,0,0,0,7,8,1',
'0,0,0,0,0,7,8,2',
'0,0,0,0,0,7,8,3',
'0,0,0,0,0,7,8,4',
'0,0,0,0,0,7,8,5',
'0,0,0,0,0,7,8,6',
'0,0,0,0,0,7,8,7',
'0,0,0,0,0,7,8,8',
'0,0,0,0,0,7,8,9',
'0,0,0,0,0,7,9,0',
'0,0,0,0,0,7,9,1',
'0,0,0,0,0,7,9,2',
'0,0,0,0,0,7,9,3',
'0,0,0,0,0,7,9,4',
'0,0,0,0,0,7,9,5',
'0,0,0,0,0,7,9,6',
'0,0,0,0,0,7,9,7',
'0,0,0,0,0,7,9,8',
'0,0,0,0,0,7,9,9',
'0,0,0,0,0,9,0,0',
'0,0,0,0,0,9,0,1',
'0,0,0,0,0,9,0,2',
'0,0,0,0,0,9,0,3',
'0,0,0,0,0,9,0,4',
'0,0,0,0,0,9,0,5',
'0,0,0,0,0,9,0,6',
'0,0,0,0,0,9,0,7',
'0,0,0,0,0,9,0,8',
'0,0,0,0,0,9,0,9',
'0,0,0,0,0,9,1,0',
'0,0,0,0,0,9,1,1',
'0,0,0,0,0,9,1,2',
'0,0,0,0,0,9,1,3',
'0,0,0,0,0,9,1,4',
'0,0,0,0,0,9,1,5',
'0,0,0,0,0,9,1,6',
'0,0,0,0,0,9,1,7',
'0,0,0,0,0,9,1,8',
'0,0,0,0,0,9,1,9',
'0,0,0,0,0,9,2,0',
'0,0,0,0,0,9,2,1',
'0,0,0,0,0,9,2,2',
'0,0,0,0,0,9,2,3',
'0,0,0,0,0,9,2,4',
'0,0,0,0,0,9,2,5',
'0,0,0,0,0,9,2,6',
'0,0,0,0,0,9,2,7',
'0,0,0,0,0,9,2,8',
'0,0,0,0,0,9,2,9',
'0,0,0,0,0,9,3,0',
'0,0,0,0,0,9,3,1',
'0,0,0,0,0,9,3,2',
'0,0,0,0,0,9,3,3',
'0,0,0,0,0,9,3,4',
'0,0,0,0,0,9,3,5',
'0,0,0,0,0,9,3,6',
'0,0,0,0,0,9,3,7',
'0,0,0,0,0,9,3,8',
'0,0,0,0,0,9,3,9',
'0,0,0,0,0,9,4,0',
'0,0,0,0,0,9,4,1',
'0,0,0,0,0,9,4,2',
'0,0,0,0,0,9,4,3',
'0,0,0,0,0,9,4,4',
'0,0,0,0,0,9,4,5',
'0,0,0,0,0,9,4,6',
'0,0,0,0,0,9,4,7',
'0,0,0,0,0,9,4,8',
'0,0,0,0,0,9,4,9',
'0,0,0,0,0,9,5,0',
'0,0,0,0,0,9,5,1',
'0,0,0,0,0,9,5,2',
'0,0,0,0,0,9,5,3',
'0,0,0,0,0,9,5,4',
'0,0,0,0,0,9,5,5',
'0,0,0,0,0,9,5,6',
'0,0,0,0,0,9,5,7',
'0,0,0,0,0,9,5,8',
'0,0,0,0,0,9,5,9',
'0,0,0,0,0,9,6,0',
'0,0,0,0,0,9,6,1',
'0,0,0,0,0,9,6,2',
'0,0,0,0,0,9,6,3',
'0,0,0,0,0,9,6,4',
'0,0,0,0,0,9,6,5',
'0,0,0,0,0,9,6,6',
'0,0,0,0,0,9,6,7',
'0,0,0,0,0,9,6,8',
'0,0,0,0,0,9,6,9',
'0,0,0,0,0,9,7,0',
'0,0,0,0,0,9,7,1',
'0,0,0,0,0,9,7,2',
'0,0,0,0,0,9,7,3',
'0,0,0,0,0,9,7,4',
'0,0,0,0,0,9,7,5',
'0,0,0,0,0,9,7,6',
'0,0,0,0,0,9,7,7',
'0,0,0,0,0,9,7,8',
'0,0,0,0,0,9,7,9',
'0,0,0,0,0,9,8,0',
'0,0,0,0,0,9,8,1',
'0,0,0,0,0,9,8,2',
'0,0,0,0,0,9,8,3',
'0,0,0,0,0,9,8,4',
'0,0,0,0,0,9,8,5',
'0,0,0,0,0,9,8,6',
'0,0,0,0,0,9,8,7',
'0,0,0,0,0,9,8,8',
'0,0,0,0,0,9,8,9',
'0,0,0,0,0,9,9,0',
'0,0,0,0,0,9,9,1',
'0,0,0,0,0,9,9,2',
'0,0,0,0,0,9,9,3',
'0,0,0,0,0,9,9,4',
'0,0,0,0,0,9,9,5',
'0,0,0,0,0,9,9,6',
'0,0,0,0,0,9,9,7',
'0,0,0,0,0,9,9,8',
'0,0,0,0,0,9,9,9',
'0,0,0,0,1,1,0,0',
'0,0,0,0,1,1,0,1',
'0,0,0,0,1,1,0,2',
'0,0,0,0,1,1,0,3',
'0,0,0,0,1,1,0,4',
'0,0,0,0,1,1,0,5',
'0,0,0,0,1,1,0,6',
'0,0,0,0,1,1,0,7',
'0,0,0,0,1,1,0,8',
'0,0,0,0,1,1,0,9',
'0,0,0,0,1,3,0,0',
'0,0,0,0,1,3,0,1',
'0,0,0,0,1,3,0,2',
'0,0,0,0,1,3,0,3',
'0,0,0,0,1,3,0,4',
'0,0,0,0,1,3,0,5',
'0,0,0,0,1,3,0,6',
'0,0,0,0,1,3,0,7',
'0,0,0,0,1,3,0,8',
'0,0,0,0,1,3,0,9',
'0,0,0,0,1,7,0,0',
'0,0,0,0,1,7,0,1',
'0,0,0,0,1,7,0,2',
'0,0,0,0,1,7,0,3',
'0,0,0,0,1,7,0,4',
'0,0,0,0,1,7,0,5',
'0,0,0,0,1,7,0,6',
'0,0,0,0,1,7,0,7',
'0,0,0,0,1,7,0,8',
'0,0,0,0,1,7,0,9',
'0,0,0,0,1,9,0,0',
'0,0,0,0,1,9,0,1',
'0,0,0,0,1,9,0,2',
'0,0,0,0,1,9,0,3',
'0,0,0,0,1,9,0,4',
'0,0,0,0,1,9,0,5',
'0,0,0,0,1,9,0,6',
'0,0,0,0,1,9,0,7',
'0,0,0,0,1,9,0,8',
'0,0,0,0,1,9,0,9',
'0,0,0,0,2,1,0,0',
'0,0,0,0,2,1,0,1',
'0,0,0,0,2,1,0,2',
'0,0,0,0,2,1,0,3',
'0,0,0,0,2,1,0,4',
'0,0,0,0,2,1,0,5',
'0,0,0,0,2,1,0,6',
'0,0,0,0,2,1,0,7',
'0,0,0,0,2,1,0,8',
'0,0,0,0,2,1,0,9',
'0,0,0,0,2,1,5,0',
'0,0,0,0,2,1,5,1',
'0,0,0,0,2,1,5,2',
'0,0,0,0,2,1,5,3',
'0,0,0,0,2,1,5,4',
'0,0,0,0,2,1,5,5',
'0,0,0,0,2,1,5,6',
'0,0,0,0,2,1,5,7',
'0,0,0,0,2,1,5,8',
'0,0,0,0,2,1,5,9',
'0,0,0,0,2,3,0,0',
'0,0,0,0,2,3,0,1',
'0,0,0,0,2,3,0,2',
'0,0,0,0,2,3,0,3',
'0,0,0,0,2,3,0,4',
'0,0,0,0,2,3,0,5',
'0,0,0,0,2,3,0,6',
'0,0,0,0,2,3,0,7',
'0,0,0,0,2,3,0,8',
'0,0,0,0,2,3,0,9',
'0,0,0,0,2,3,5,0',
'0,0,0,0,2,3,5,1',
'0,0,0,0,2,3,5,2',
'0,0,0,0,2,3,5,3',
'0,0,0,0,2,3,5,4',
'0,0,0,0,2,3,5,5',
'0,0,0,0,2,3,5,6',
'0,0,0,0,2,3,5,7',
'0,0,0,0,2,3,5,8',
'0,0,0,0,2,3,5,9',
'0,0,0,0,2,7,0,0',
'0,0,0,0,2,7,0,1',
'0,0,0,0,2,7,0,2',
'0,0,0,0,2,7,0,3',
'0,0,0,0,2,7,0,4',
'0,0,0,0,2,7,0,5',
'0,0,0,0,2,7,0,6',
'0,0,0,0,2,7,0,7',
'0,0,0,0,2,7,0,8',
'0,0,0,0,2,7,0,9',
'0,0,0,0,2,7,5,0',
'0,0,0,0,2,7,5,1',
'0,0,0,0,2,7,5,2',
'0,0,0,0,2,7,5,3',
'0,0,0,0,2,7,5,4',
'0,0,0,0,2,7,5,5',
'0,0,0,0,2,7,5,6',
'0,0,0,0,2,7,5,7',
'0,0,0,0,2,7,5,8',
'0,0,0,0,2,7,5,9',
'0,0,0,0,2,9,0,0',
'0,0,0,0,2,9,0,1',
'0,0,0,0,2,9,0,2',
'0,0,0,0,2,9,0,3',
'0,0,0,0,2,9,0,4',
'0,0,0,0,2,9,0,5',
'0,0,0,0,2,9,0,6',
'0,0,0,0,2,9,0,7',
'0,0,0,0,2,9,0,8',
'0,0,0,0,2,9,0,9',
'0,0,0,0,2,9,5,0',
'0,0,0,0,2,9,5,1',
'0,0,0,0,2,9,5,2',
'0,0,0,0,2,9,5,3',
'0,0,0,0,2,9,5,4',
'0,0,0,0,2,9,5,5',
'0,0,0,0,2,9,5,6',
'0,0,0,0,2,9,5,7',
'0,0,0,0,2,9,5,8',
'0,0,0,0,2,9,5,9',
'0,0,0,0,3,1,0,0',
'0,0,0,0,3,1,0,1',
'0,0,0,0,3,1,0,2',
'0,0,0,0,3,1,0,3',
'0,0,0,0,3,1,0,4',
'0,0,0,0,3,1,0,5',
'0,0,0,0,3,1,0,6',
'0,0,0,0,3,1,0,7',
'0,0,0,0,3,1,0,8',
'0,0,0,0,3,1,0,9',
'0,0,0,0,3,3,0,0',
'0,0,0,0,3,3,0,1',
'0,0,0,0,3,3,0,2',
'0,0,0,0,3,3,0,3',
'0,0,0,0,3,3,0,4',
'0,0,0,0,3,3,0,5',
'0,0,0,0,3,3,0,6',
'0,0,0,0,3,3,0,7',
'0,0,0,0,3,3,0,8',
'0,0,0,0,3,3,0,9',
'0,0,0,0,3,7,0,0',
'0,0,0,0,3,7,0,1',
'0,0,0,0,3,7,0,2',
'0,0,0,0,3,7,0,3',
'0,0,0,0,3,7,0,4',
'0,0,0,0,3,7,0,5',
'0,0,0,0,3,7,0,6',
'0,0,0,0,3,7,0,7',
'0,0,0,0,3,7,0,8',
'0,0,0,0,3,7,0,9',
'0,0,0,0,3,9,0,0',
'0,0,0,0,3,9,0,1',
'0,0,0,0,3,9,0,2',
'0,0,0,0,3,9,0,3',
'0,0,0,0,3,9,0,4',
'0,0,0,0,3,9,0,5',
'0,0,0,0,3,9,0,6',
'0,0,0,0,3,9,0,7',
'0,0,0,0,3,9,0,8',
'0,0,0,0,3,9,0,9',
'0,0,0,0,4,1,0,0',
'0,0,0,0,4,1,0,1',
'0,0,0,0,4,1,0,2',
'0,0,0,0,4,1,0,3',
'0,0,0,0,4,1,0,4',
'0,0,0,0,4,1,0,5',
'0,0,0,0,4,1,0,6',
'0,0,0,0,4,1,0,7',
'0,0,0,0,4,1,0,8',
'0,0,0,0,4,1,0,9',
'0,0,0,0,4,1,5,0',
'0,0,0,0,4,1,5,1',
'0,0,0,0,4,1,5,2',
'0,0,0,0,4,1,5,3',
'0,0,0,0,4,1,5,4',
'0,0,0,0,4,1,5,5',
'0,0,0,0,4,1,5,6',
'0,0,0,0,4,1,5,7',
'0,0,0,0,4,1,5,8',
'0,0,0,0,4,1,5,9',
'0,0,0,0,4,3,0,0',
'0,0,0,0,4,3,0,1',
'0,0,0,0,4,3,0,2',
'0,0,0,0,4,3,0,3',
'0,0,0,0,4,3,0,4',
'0,0,0,0,4,3,0,5',
'0,0,0,0,4,3,0,6',
'0,0,0,0,4,3,0,7',
'0,0,0,0,4,3,0,8',
'0,0,0,0,4,3,0,9',
'0,0,0,0,4,3,5,0',
'0,0,0,0,4,3,5,1',
'0,0,0,0,4,3,5,2',
'0,0,0,0,4,3,5,3',
'0,0,0,0,4,3,5,4',
'0,0,0,0,4,3,5,5',
'0,0,0,0,4,3,5,6',
'0,0,0,0,4,3,5,7',
'0,0,0,0,4,3,5,8',
'0,0,0,0,4,3,5,9',
'0,0,0,0,4,7,0,0',
'0,0,0,0,4,7,0,1',
'0,0,0,0,4,7,0,2',
'0,0,0,0,4,7,0,3',
'0,0,0,0,4,7,0,4',
'0,0,0,0,4,7,0,5',
'0,0,0,0,4,7,0,6',
'0,0,0,0,4,7,0,7',
'0,0,0,0,4,7,0,8',
'0,0,0,0,4,7,0,9',
'0,0,0,0,4,7,5,0',
'0,0,0,0,4,7,5,1',
'0,0,0,0,4,7,5,2',
'0,0,0,0,4,7,5,3',
'0,0,0,0,4,7,5,4',
'0,0,0,0,4,7,5,5',
'0,0,0,0,4,7,5,6',
'0,0,0,0,4,7,5,7',
'0,0,0,0,4,7,5,8',
'0,0,0,0,4,7,5,9',
'0,0,0,0,4,9,0,0',
'0,0,0,0,4,9,0,1',
'0,0,0,0,4,9,0,2',
'0,0,0,0,4,9,0,3',
'0,0,0,0,4,9,0,4',
'0,0,0,0,4,9,0,5',
'0,0,0,0,4,9,0,6',
'0,0,0,0,4,9,0,7',
'0,0,0,0,4,9,0,8',
'0,0,0,0,4,9,0,9',
'0,0,0,0,4,9,5,0',
'0,0,0,0,4,9,5,1',
'0,0,0,0,4,9,5,2',
'0,0,0,0,4,9,5,3',
'0,0,0,0,4,9,5,4',
'0,0,0,0,4,9,5,5',
'0,0,0,0,4,9,5,6',
'0,0,0,0,4,9,5,7',
'0,0,0,0,4,9,5,8',
'0,0,0,0,4,9,5,9',
'0,0,0,0,5,1,0,0',
'0,0,0,0,5,1,0,1',
'0,0,0,0,5,1,0,2',
'0,0,0,0,5,1,0,3',
'0,0,0,0,5,1,0,4',
'0,0,0,0,5,1,0,5',
'0,0,0,0,5,1,0,6',
'0,0,0,0,5,1,0,7',
'0,0,0,0,5,1,0,8',
'0,0,0,0,5,1,0,9',
'0,0,0,0,5,1,2,0',
'0,0,0,0,5,1,2,1',
'0,0,0,0,5,1,2,2',
'0,0,0,0,5,1,2,3',
'0,0,0,0,5,1,2,4',
'0,0,0,0,5,1,2,5',
'0,0,0,0,5,1,2,6',
'0,0,0,0,5,1,2,7',
'0,0,0,0,5,1,2,8',
'0,0,0,0,5,1,2,9',
'0,0,0,0,5,1,4,0',
'0,0,0,0,5,1,4,1',
'0,0,0,0,5,1,4,2',
'0,0,0,0,5,1,4,3',
'0,0,0,0,5,1,4,4',
'0,0,0,0,5,1,4,5',
'0,0,0,0,5,1,4,6',
'0,0,0,0,5,1,4,7',
'0,0,0,0,5,1,4,8',
'0,0,0,0,5,1,4,9',
'0,0,0,0,5,1,6,0',
'0,0,0,0,5,1,6,1',
'0,0,0,0,5,1,6,2',
'0,0,0,0,5,1,6,3',
'0,0,0,0,5,1,6,4',
'0,0,0,0,5,1,6,5',
'0,0,0,0,5,1,6,6',
'0,0,0,0,5,1,6,7',
'0,0,0,0,5,1,6,8',
'0,0,0,0,5,1,6,9',
'0,0,0,0,5,1,8,0',
'0,0,0,0,5,1,8,1',
'0,0,0,0,5,1,8,2',
'0,0,0,0,5,1,8,3',
'0,0,0,0,5,1,8,4',
'0,0,0,0,5,1,8,5',
'0,0,0,0,5,1,8,6',
'0,0,0,0,5,1,8,7',
'0,0,0,0,5,1,8,8',
'0,0,0,0,5,1,8,9',
'0,0,0,0,5,3,0,0',
'0,0,0,0,5,3,0,1',
'0,0,0,0,5,3,0,2',
'0,0,0,0,5,3,0,3',
'0,0,0,0,5,3,0,4',
'0,0,0,0,5,3,0,5',
'0,0,0,0,5,3,0,6',
'0,0,0,0,5,3,0,7',
'0,0,0,0,5,3,0,8',
'0,0,0,0,5,3,0,9',
'0,0,0,0,5,3,2,0',
'0,0,0,0,5,3,2,1',
'0,0,0,0,5,3,2,2',
'0,0,0,0,5,3,2,3',
'0,0,0,0,5,3,2,4',
'0,0,0,0,5,3,2,5',
'0,0,0,0,5,3,2,6',
'0,0,0,0,5,3,2,7',
'0,0,0,0,5,3,2,8',
'0,0,0,0,5,3,2,9',
'0,0,0,0,5,3,4,0',
'0,0,0,0,5,3,4,1',
'0,0,0,0,5,3,4,2',
'0,0,0,0,5,3,4,3',
'0,0,0,0,5,3,4,4',
'0,0,0,0,5,3,4,5',
'0,0,0,0,5,3,4,6',
'0,0,0,0,5,3,4,7',
'0,0,0,0,5,3,4,8',
'0,0,0,0,5,3,4,9',
'0,0,0,0,5,3,6,0',
'0,0,0,0,5,3,6,1',
'0,0,0,0,5,3,6,2',
'0,0,0,0,5,3,6,3',
'0,0,0,0,5,3,6,4',
'0,0,0,0,5,3,6,5',
'0,0,0,0,5,3,6,6',
'0,0,0,0,5,3,6,7',
'0,0,0,0,5,3,6,8',
'0,0,0,0,5,3,6,9',
'0,0,0,0,5,3,8,0',
'0,0,0,0,5,3,8,1',
'0,0,0,0,5,3,8,2',
'0,0,0,0,5,3,8,3',
'0,0,0,0,5,3,8,4',
'0,0,0,0,5,3,8,5',
'0,0,0,0,5,3,8,6',
'0,0,0,0,5,3,8,7',
'0,0,0,0,5,3,8,8',
'0,0,0,0,5,3,8,9',
'0,0,0,0,5,7,0,0',
'0,0,0,0,5,7,0,1',
'0,0,0,0,5,7,0,2',
'0,0,0,0,5,7,0,3',
'0,0,0,0,5,7,0,4',
'0,0,0,0,5,7,0,5',
'0,0,0,0,5,7,0,6',
'0,0,0,0,5,7,0,7',
'0,0,0,0,5,7,0,8',
'0,0,0,0,5,7,0,9',
'0,0,0,0,5,7,2,0',
'0,0,0,0,5,7,2,1',
'0,0,0,0,5,7,2,2',
'0,0,0,0,5,7,2,3',
'0,0,0,0,5,7,2,4',
'0,0,0,0,5,7,2,5',
'0,0,0,0,5,7,2,6',
'0,0,0,0,5,7,2,7',
'0,0,0,0,5,7,2,8',
'0,0,0,0,5,7,2,9',
'0,0,0,0,5,7,4,0',
'0,0,0,0,5,7,4,1',
'0,0,0,0,5,7,4,2',
'0,0,0,0,5,7,4,3',
'0,0,0,0,5,7,4,4',
'0,0,0,0,5,7,4,5',
'0,0,0,0,5,7,4,6',
'0,0,0,0,5,7,4,7',
'0,0,0,0,5,7,4,8',
'0,0,0,0,5,7,4,9',
'0,0,0,0,5,7,6,0',
'0,0,0,0,5,7,6,1',
'0,0,0,0,5,7,6,2',
'0,0,0,0,5,7,6,3',
'0,0,0,0,5,7,6,4',
'0,0,0,0,5,7,6,5',
'0,0,0,0,5,7,6,6',
'0,0,0,0,5,7,6,7',
'0,0,0,0,5,7,6,8',
'0,0,0,0,5,7,6,9',
'0,0,0,0,5,7,8,0',
'0,0,0,0,5,7,8,1',
'0,0,0,0,5,7,8,2',
'0,0,0,0,5,7,8,3',
'0,0,0,0,5,7,8,4',
'0,0,0,0,5,7,8,5',
'0,0,0,0,5,7,8,6',
'0,0,0,0,5,7,8,7',
'0,0,0,0,5,7,8,8',
'0,0,0,0,5,7,8,9',
'0,0,0,0,5,9,0,0',
'0,0,0,0,5,9,0,1',
'0,0,0,0,5,9,0,2',
'0,0,0,0,5,9,0,3',
'0,0,0,0,5,9,0,4',
'0,0,0,0,5,9,0,5',
'0,0,0,0,5,9,0,6',
'0,0,0,0,5,9,0,7',
'0,0,0,0,5,9,0,8',
'0,0,0,0,5,9,0,9',
'0,0,0,0,5,9,2,0',
'0,0,0,0,5,9,2,1',
'0,0,0,0,5,9,2,2',
'0,0,0,0,5,9,2,3',
'0,0,0,0,5,9,2,4',
'0,0,0,0,5,9,2,5',
'0,0,0,0,5,9,2,6',
'0,0,0,0,5,9,2,7',
'0,0,0,0,5,9,2,8',
'0,0,0,0,5,9,2,9',
'0,0,0,0,5,9,4,0',
'0,0,0,0,5,9,4,1',
'0,0,0,0,5,9,4,2',
'0,0,0,0,5,9,4,3',
'0,0,0,0,5,9,4,4',
'0,0,0,0,5,9,4,5',
'0,0,0,0,5,9,4,6',
'0,0,0,0,5,9,4,7',
'0,0,0,0,5,9,4,8',
'0,0,0,0,5,9,4,9',
'0,0,0,0,5,9,6,0',
'0,0,0,0,5,9,6,1',
'0,0,0,0,5,9,6,2',
'0,0,0,0,5,9,6,3',
'0,0,0,0,5,9,6,4',
'0,0,0,0,5,9,6,5',
'0,0,0,0,5,9,6,6',
'0,0,0,0,5,9,6,7',
'0,0,0,0,5,9,6,8',
'0,0,0,0,5,9,6,9',
'0,0,0,0,5,9,8,0',
'0,0,0,0,5,9,8,1',
'0,0,0,0,5,9,8,2',
'0,0,0,0,5,9,8,3',
'0,0,0,0,5,9,8,4',
'0,0,0,0,5,9,8,5',
'0,0,0,0,5,9,8,6',
'0,0,0,0,5,9,8,7',
'0,0,0,0,5,9,8,8',
'0,0,0,0,5,9,8,9',
'0,0,0,0,6,1,0,0',
'0,0,0,0,6,1,0,1',
'0,0,0,0,6,1,0,2',
'0,0,0,0,6,1,0,3',
'0,0,0,0,6,1,0,4',
'0,0,0,0,6,1,0,5',
'0,0,0,0,6,1,0,6',
'0,0,0,0,6,1,0,7',
'0,0,0,0,6,1,0,8',
'0,0,0,0,6,1,0,9',
'0,0,0,0,6,1,5,0',
'0,0,0,0,6,1,5,1',
'0,0,0,0,6,1,5,2',
'0,0,0,0,6,1,5,3',
'0,0,0,0,6,1,5,4',
'0,0,0,0,6,1,5,5',
'0,0,0,0,6,1,5,6',
'0,0,0,0,6,1,5,7',
'0,0,0,0,6,1,5,8',
'0,0,0,0,6,1,5,9',
'0,0,0,0,6,3,0,0',
'0,0,0,0,6,3,0,1',
'0,0,0,0,6,3,0,2',
'0,0,0,0,6,3,0,3',
'0,0,0,0,6,3,0,4',
'0,0,0,0,6,3,0,5',
'0,0,0,0,6,3,0,6',
'0,0,0,0,6,3,0,7',
'0,0,0,0,6,3,0,8',
'0,0,0,0,6,3,0,9',
'0,0,0,0,6,3,5,0',
'0,0,0,0,6,3,5,1',
'0,0,0,0,6,3,5,2',
'0,0,0,0,6,3,5,3',
'0,0,0,0,6,3,5,4',
'0,0,0,0,6,3,5,5',
'0,0,0,0,6,3,5,6',
'0,0,0,0,6,3,5,7',
'0,0,0,0,6,3,5,8',
'0,0,0,0,6,3,5,9',
'0,0,0,0,6,7,0,0',
'0,0,0,0,6,7,0,1',
'0,0,0,0,6,7,0,2',
'0,0,0,0,6,7,0,3',
'0,0,0,0,6,7,0,4',
'0,0,0,0,6,7,0,5',
'0,0,0,0,6,7,0,6',
'0,0,0,0,6,7,0,7',
'0,0,0,0,6,7,0,8',
'0,0,0,0,6,7,0,9',
'0,0,0,0,6,7,5,0',
'0,0,0,0,6,7,5,1',
'0,0,0,0,6,7,5,2',
'0,0,0,0,6,7,5,3',
'0,0,0,0,6,7,5,4',
'0,0,0,0,6,7,5,5',
'0,0,0,0,6,7,5,6',
'0,0,0,0,6,7,5,7',
'0,0,0,0,6,7,5,8',
'0,0,0,0,6,7,5,9',
'0,0,0,0,6,9,0,0',
'0,0,0,0,6,9,0,1',
'0,0,0,0,6,9,0,2',
'0,0,0,0,6,9,0,3',
'0,0,0,0,6,9,0,4',
'0,0,0,0,6,9,0,5',
'0,0,0,0,6,9,0,6',
'0,0,0,0,6,9,0,7',
'0,0,0,0,6,9,0,8',
'0,0,0,0,6,9,0,9',
'0,0,0,0,6,9,5,0',
'0,0,0,0,6,9,5,1',
'0,0,0,0,6,9,5,2',
'0,0,0,0,6,9,5,3',
'0,0,0,0,6,9,5,4',
'0,0,0,0,6,9,5,5',
'0,0,0,0,6,9,5,6',
'0,0,0,0,6,9,5,7',
'0,0,0,0,6,9,5,8',
'0,0,0,0,6,9,5,9',
'0,0,0,0,7,1,0,0',
'0,0,0,0,7,1,0,1',
'0,0,0,0,7,1,0,2',
'0,0,0,0,7,1,0,3',
'0,0,0,0,7,1,0,4',
'0,0,0,0,7,1,0,5',
'0,0,0,0,7,1,0,6',
'0,0,0,0,7,1,0,7',
'0,0,0,0,7,1,0,8',
'0,0,0,0,7,1,0,9',
'0,0,0,0,7,3,0,0',
'0,0,0,0,7,3,0,1',
'0,0,0,0,7,3,0,2',
'0,0,0,0,7,3,0,3',
'0,0,0,0,7,3,0,4',
'0,0,0,0,7,3,0,5',
'0,0,0,0,7,3,0,6',
'0,0,0,0,7,3,0,7',
'0,0,0,0,7,3,0,8',
'0,0,0,0,7,3,0,9',
'0,0,0,0,7,7,0,0',
'0,0,0,0,7,7,0,1',
'0,0,0,0,7,7,0,2',
'0,0,0,0,7,7,0,3',
'0,0,0,0,7,7,0,4',
'0,0,0,0,7,7,0,5',
'0,0,0,0,7,7,0,6',
'0,0,0,0,7,7,0,7',
'0,0,0,0,7,7,0,8',
'0,0,0,0,7,7,0,9',
'0,0,0,0,7,9,0,0',
'0,0,0,0,7,9,0,1',
'0,0,0,0,7,9,0,2',
'0,0,0,0,7,9,0,3',
'0,0,0,0,7,9,0,4',
'0,0,0,0,7,9,0,5',
'0,0,0,0,7,9,0,6',
'0,0,0,0,7,9,0,7',
'0,0,0,0,7,9,0,8',
'0,0,0,0,7,9,0,9',
'0,0,0,0,8,1,0,0',
'0,0,0,0,8,1,0,1',
'0,0,0,0,8,1,0,2',
'0,0,0,0,8,1,0,3',
'0,0,0,0,8,1,0,4',
'0,0,0,0,8,1,0,5',
'0,0,0,0,8,1,0,6',
'0,0,0,0,8,1,0,7',
'0,0,0,0,8,1,0,8',
'0,0,0,0,8,1,0,9',
'0,0,0,0,8,1,5,0',
'0,0,0,0,8,1,5,1',
'0,0,0,0,8,1,5,2',
'0,0,0,0,8,1,5,3',
'0,0,0,0,8,1,5,4',
'0,0,0,0,8,1,5,5',
'0,0,0,0,8,1,5,6',
'0,0,0,0,8,1,5,7',
'0,0,0,0,8,1,5,8',
'0,0,0,0,8,1,5,9',
'0,0,0,0,8,3,0,0',
'0,0,0,0,8,3,0,1',
'0,0,0,0,8,3,0,2',
'0,0,0,0,8,3,0,3',
'0,0,0,0,8,3,0,4',
'0,0,0,0,8,3,0,5',
'0,0,0,0,8,3,0,6',
'0,0,0,0,8,3,0,7',
'0,0,0,0,8,3,0,8',
'0,0,0,0,8,3,0,9',
'0,0,0,0,8,3,5,0',
'0,0,0,0,8,3,5,1',
'0,0,0,0,8,3,5,2',
'0,0,0,0,8,3,5,3',
'0,0,0,0,8,3,5,4',
'0,0,0,0,8,3,5,5',
'0,0,0,0,8,3,5,6',
'0,0,0,0,8,3,5,7',
'0,0,0,0,8,3,5,8',
'0,0,0,0,8,3,5,9',
'0,0,0,0,8,7,0,0',
'0,0,0,0,8,7,0,1',
'0,0,0,0,8,7,0,2',
'0,0,0,0,8,7,0,3',
'0,0,0,0,8,7,0,4',
'0,0,0,0,8,7,0,5',
'0,0,0,0,8,7,0,6',
'0,0,0,0,8,7,0,7',
'0,0,0,0,8,7,0,8',
'0,0,0,0,8,7,0,9',
'0,0,0,0,8,7,5,0',
'0,0,0,0,8,7,5,1',
'0,0,0,0,8,7,5,2',
'0,0,0,0,8,7,5,3',
'0,0,0,0,8,7,5,4',
'0,0,0,0,8,7,5,5',
'0,0,0,0,8,7,5,6',
'0,0,0,0,8,7,5,7',
'0,0,0,0,8,7,5,8',
'0,0,0,0,8,7,5,9',
'0,0,0,0,8,9,0,0',
'0,0,0,0,8,9,0,1',
'0,0,0,0,8,9,0,2',
'0,0,0,0,8,9,0,3',
'0,0,0,0,8,9,0,4',
'0,0,0,0,8,9,0,5',
'0,0,0,0,8,9,0,6',
'0,0,0,0,8,9,0,7',
'0,0,0,0,8,9,0,8',
'0,0,0,0,8,9,0,9',
'0,0,0,0,8,9,5,0',
'0,0,0,0,8,9,5,1',
'0,0,0,0,8,9,5,2',
'0,0,0,0,8,9,5,3',
'0,0,0,0,8,9,5,4',
'0,0,0,0,8,9,5,5',
'0,0,0,0,8,9,5,6',
'0,0,0,0,8,9,5,7',
'0,0,0,0,8,9,5,8',
'0,0,0,0,8,9,5,9',
'0,0,0,0,9,1,0,0',
'0,0,0,0,9,1,0,1',
'0,0,0,0,9,1,0,2',
'0,0,0,0,9,1,0,3',
'0,0,0,0,9,1,0,4',
'0,0,0,0,9,1,0,5',
'0,0,0,0,9,1,0,6',
'0,0,0,0,9,1,0,7',
'0,0,0,0,9,1,0,8',
'0,0,0,0,9,1,0,9',
'0,0,0,0,9,3,0,0',
'0,0,0,0,9,3,0,1',
'0,0,0,0,9,3,0,2',
'0,0,0,0,9,3,0,3',
'0,0,0,0,9,3,0,4',
'0,0,0,0,9,3,0,5',
'0,0,0,0,9,3,0,6',
'0,0,0,0,9,3,0,7',
'0,0,0,0,9,3,0,8',
'0,0,0,0,9,3,0,9',
'0,0,0,0,9,7,0,0',
'0,0,0,0,9,7,0,1',
'0,0,0,0,9,7,0,2',
'0,0,0,0,9,7,0,3',
'0,0,0,0,9,7,0,4',
'0,0,0,0,9,7,0,5',
'0,0,0,0,9,7,0,6',
'0,0,0,0,9,7,0,7',
'0,0,0,0,9,7,0,8',
'0,0,0,0,9,7,0,9',
'0,0,0,0,9,9,0,0',
'0,0,0,0,9,9,0,1',
'0,0,0,0,9,9,0,2',
'0,0,0,0,9,9,0,3',
'0,0,0,0,9,9,0,4',
'0,0,0,0,9,9,0,5',
'0,0,0,0,9,9,0,6',
'0,0,0,0,9,9,0,7',
'0,0,0,0,9,9,0,8',
'0,0,0,0,9,9,0,9']

"""#output"""

num_clusters=2
min_split_size=3
max_split_size=8
count=100
while(count):
    cellular_automata_clustering(rule_list,num_of_splits, split_enc,n,num_clusters,min_split_size,max_split_size)
    count=count-1