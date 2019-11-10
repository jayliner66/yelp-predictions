import json

def getDate(date):
    d = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    year = int(date[0:4])
    month = int(date[5:7])
    date = int(date[8:10])
    bucket = (year-2013)*365 + date
    if(month > 1):
        bucket += d[month-2]
    bucket //= 7

    return bucket
def json_readline(file):
    for line in open(file, mode="r"):
        yield json.loads(line)

business_dict = {}
for business in json_readline("yelp_dataset/business.json"):
    business["reviews"] = [0 for i in range(0, 52*7)]

    id = business["business_id"]

    business_dict[id] = business

review_dict = {}
counter = 0
for review in json_readline("yelp_dataset/review.json"):
    review_dict[review["review_id"]] = review
    review_time = getDate(review["date"])
    if(review_time < 0):
        continue
    if review["business_id"] in business_dict:
        business_dict[review["business_id"]]["reviews"][review_time] += 1
    counter += 1
    if(counter % 1000 == 0):
        print(counter)

with open('business_reviewcounts.json', 'w') as fp:
    json.dump(business_dict, fp)
