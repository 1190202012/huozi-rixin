import json

a = json.load(open("result.json", "r", encoding= "UTF-8"))

for i in a.values():
    i["details"] = {
        "no knowledge response": i["no knowledge response"],
        "summarize google good": i["summarize google good"],
        "summarize google plus wiki": i["summarize google plus wiki"],
        "contrive google no truncate merged": i["contrive google no truncate merged"],
        "gen doc": i["gen doc"]
    }
    
    i["details"]["no knowledge response"]["response"] = i["details"]["no knowledge response"]["response"][:200]
        
    i.pop("summarize google good")
    i.pop("summarize google plus wiki")
    i.pop("contrive google no truncate merged")
    i.pop("gen doc")
    i.pop("no knowledge response")

json.dump(a, open("result3.json","w",encoding="UTF-8"), ensure_ascii=False, indent=4)