from datasets import concatenate_datasets, load_from_disk

ds1 = load_from_disk("data/YES_Mask1_merged")
ds2 = load_from_disk("data/YES_Mask2_merged")
merged = concatenate_datasets([ds1, ds2])
merged.save_to_disk("data/YES_Mask12_merged")
print("合并完成，样本数量：", len(merged))