
def json_readline(file):
    for line in open(file, mode="r"):
        yield json.loads(line)

def main():
    review_dict = {}
    for review in json_readline("yelp_dataset/review.json"):
        review["embedding"] = getEmbedding(review["text"])
        review_dict[review["review_id"]] = review


main()
