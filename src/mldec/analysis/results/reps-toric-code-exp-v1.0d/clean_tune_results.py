import os
import json


dirs = [f"zjob_{i}" for i in range(50)]
temp_best = []
not_done = 0
for i, dir in enumerate(dirs):
    # print(f"dir: {dir}")
    hyper_file = "hyper_config.json"
    # check if hyper file is in dir, if not the job is not done
    if not os.path.exists(os.path.join(dir, hyper_file)):
        not_done += 1
        continue

    # open dir/tune_results.csv
    tune_results_file = os.path.join(dir, "tune_results.csv")

    file = open(tune_results_file)
    lines = file.readlines()
    cycle = len(lines[0].split(","))
    compare_index = 5
    new_lines = []
    new = ""
    counter = 1
    for s in lines[1].split(","):
        new += s
        if counter == cycle:
            new += "\n"
            new_lines.append(new)
            new = ""
            counter = 0
        else:
            new += ","
        counter += 1
        # comparison for aggregation

    new_file = os.path.join(dir, "new_tune_results.csv")
    # delete the existing file
    if os.path.exists(new_file):
        os.remove(new_file)
    new_file = open(new_file, "w")
    new_file.write(lines[0] + "".join(new_lines))
    new_file.close()

    best_val_acc = 0
    best_index = 0
    for k, newline in enumerate(new_lines):
        val_acc = float(newline.split(",")[compare_index])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_index = k
    # write single line to temp_best
    dct = json.load(open(os.path.join(dir, hyper_file)))
    temp_header = ",".join([str(k) for k in dct.keys()]) +"," + lines[0]
    hypers = ""
    for kv, v in enumerate(dct.values()):
        if kv == 0:
            hypers+= str(float(v)) + ","
        else:
            hypers += str(int(v)) + ","
    temp_best.append(hypers + new_lines[best_index])

    file.close()


new_temp_best_file = "temp_results.csv"
if os.path.exists(new_temp_best_file):
    # delete the existing file
    os.remove(new_temp_best_file)
temp_file = open(new_temp_best_file, "w")
temp_file.write(temp_header + "".join(temp_best))
temp_file.close()

print("files not done: ", not_done)

            


