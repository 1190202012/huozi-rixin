import os
import json

def collect_saved_files(output_dir, world_size):
    output_dir = os.path.normpath(output_dir)
    cached_dir = output_dir.replace("$rank$", "")
        
    for file in os.listdir(output_dir.replace("$rank$", "0")):
        if file[-4:] != "json":
            continue
        result = {}
        for i in range(world_size): 
            rank_path = output_dir.replace("$rank$", str(i))
            if not os.path.exists(os.path.join(rank_path, file)):
                print(f"不存在{os.path.join(rank_path, file)}")
                continue
            part_result = json.load(open(os.path.join(rank_path, file), "r", encoding="UTF-8"))

            if "active_methods" in part_result:
                if i == 0:
                    result = part_result
                else:
                    result["processed_knowledge"].update(part_result["processed_knowledge"])

            else:
                result.update(part_result)
        
        json.dump(result, open(os.path.join(cached_dir, file) , "w", encoding="UTF-8"), ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # path = os.path.join(os.path.abspath(output_dir), "$rank$")
    world_rank = 6
    path = "/home/yfchen/retrieval-demo/output/open_natural_question/2023-11-10-21-25-01/$rank$"
    collect_saved_files(path, world_rank)