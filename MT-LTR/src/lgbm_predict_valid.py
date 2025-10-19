# encoding=utf8
import sys

def cal_one_group(group):
    sorted_group = [item[0] for item in sorted(group, key=lambda x: x[1], reverse=True)]
    return sum(sorted_group[:1]), sum(sorted_group[:3]), sum(sorted_group[:5])

startFile = sys.argv[1]
endFile = sys.argv[2]
country_code = sys.argv[3]

#分母
total_group = 0
# top1/3/5 click
total_hit_top1 = 0
total_hit_top3 = 0
total_hit_top5 = 0

last_group_id = 'last_group_id'
group = []
for l in open(startFile):
    cur_group_id, label, pred = l.strip().split('\t')
    if cur_group_id != last_group_id:
        if len(group) != 0:
            hit_top1, hit_top3, hit_top5 = cal_one_group(group)
            total_hit_top1 += hit_top1
            total_hit_top3 += hit_top3
            total_hit_top5 += hit_top5
            total_group += 1
            if total_group % 100000 == 0:
                print(f"start groups {total_group}")
            group = []
        last_group_id = cur_group_id
    group.append( [int(label) - 1, float(pred)] )

hit_top1, hit_top3, hit_top5 = cal_one_group(group)
total_hit_top1 += hit_top1
total_hit_top3 += hit_top3
total_hit_top5 += hit_top5
total_group += 1

print('%s start group: %s'%(country_code, total_group))
print('%s start top1 hit rate:%.4f'%(country_code, total_hit_top1*1.0 / total_group))
print('%s start top3 hit rate:%.4f'%(country_code, total_hit_top3*1.0 / total_group))
print('%s start top5 hit rate:%.4f'%(country_code, total_hit_top5*1.0 / total_group))

#分母
total_group_end = 0
# top1/3/5 click
total_hit_top1_end = 0
total_hit_top3_end = 0
total_hit_top5_end = 0

last_group_id = 'last_group_id'
group = []
for l in open(endFile):
    cur_group_id, label, pred = l.strip().split('\t')
    if cur_group_id != last_group_id:
        if len(group) != 0:
            hit_top1, hit_top3, hit_top5 = cal_one_group(group)
            total_hit_top1_end += hit_top1
            total_hit_top3_end += hit_top3
            total_hit_top5_end += hit_top5
            total_group_end += 1
            if total_group_end % 100000 == 0:
                print(f"end groups {total_group}")
            group = []
        last_group_id = cur_group_id
    group.append( [int(label) - 1, float(pred)] )

hit_top1, hit_top3, hit_top5 = cal_one_group(group)
total_hit_top1_end += hit_top1
total_hit_top3_end += hit_top3
total_hit_top5_end += hit_top5
total_group_end += 1

print('%s start group: %s'%(country_code, total_group_end))
print('%s start top1 hit rate:%.4f'%(country_code, total_hit_top1_end*1.0 / total_group_end))
print('%s start top3 hit rate:%.4f'%(country_code, total_hit_top3_end*1.0 / total_group_end))
print('%s start top5 hit rate:%.4f'%(country_code, total_hit_top5_end*1.0 / total_group_end))

print('%s total group: %s'%(country_code, total_group_end + total_group))
print('%s total top1 hit rate:%.4f'%(country_code, (total_hit_top1_end + total_hit_top1)*1.0 / (total_group_end+total_group)))
print('%s total top3 hit rate:%.4f'%(country_code, (total_hit_top3_end + total_hit_top3)*1.0 / (total_group_end+total_group)))
print('%s total top5 hit rate:%.4f'%(country_code, (total_hit_top5_end + total_hit_top5)*1.0 / (total_group_end+total_group)))