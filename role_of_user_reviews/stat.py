import os
from collections import Counter

def main():
    labels = []
    apps = ["Spotify", "Instagram"]
    for app in apps:
        path = "role_of_user_reviews/"+app
        files = os.listdir(path)
        for file in files:
            file_handle = open(path+"/"+file, "r", encoding="UTF-8")
            for line in file_handle:
                try:
                    label = line.split("&")[0]
                    labels.append(label)
                except Exception as e:
                    print(file + " is an empty file")
    
    res = Counter(labels)
    print(res)
    

if __name__ == '__main__':
    main()