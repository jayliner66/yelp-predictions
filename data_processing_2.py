import json

with open('business_reviewcounts.json') as json_file:
    business_dict = json.load(json_file)

large_business_dict = {}
count = 0
for id in business_dict:
    if business_dict[id]["review_count"] >= 100:
        large_business_dict[id] = business_dict[id]
        count += 1

sample = {}
count = 0
for i in large_business_dict:
    sample[i] = large_business_dict[i]
    count += 1
    if(count >= 500):
        break
with open('business_reviewcounts_large.json', 'w') as fp:
    json.dump(large_business_dict, fp)
with open('business_reviewcounts_500sample.json', 'w') as fp:
    json.dump(sample, fp)

print(count)
