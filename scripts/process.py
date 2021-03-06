
import os


def get_tokens():
    tokens_filename = "/home/wangdq/data/tokens.txt"
    with open(tokens_filename) as f:
        tokens = [t.strip() for t in f.readlines()]
    return tokens

def split_by_tokens():
    src_filename = "/home/wangdq/data/all.en"
    tgt_filename = "/home/wangdq/data/all.zh"
    src_lang, tgt_lang = 'en', 'zh'
    tokens = get_tokens()

    tokens_dict = {t:[[], []] for t in tokens}
    with open(src_filename) as f_src, open(tgt_filename) as f_tgt:
        for src, tgt in zip(f_src, f_tgt):
            src_tokens = set(src.strip().split(' '))
            for t in tokens:
                if t in src_tokens:
                    tokens_dict[t][0].append(src)
                    tokens_dict[t][1].append(tgt)

    dirname = "/home/wangdq/data/token_dataset"
    for tokens, (src, tgt) in tokens_dict.items():
        with open(os.path.join(dirname, tokens + '.' + src_lang), 'w') as f:
            f.writelines(src)
        with open(os.path.join(dirname, tokens + '.' + tgt_lang), 'w') as f:
            f.writelines(tgt)
        with open(os.path.join(dirname, tokens + '.txt'), 'w') as f:
            for s, t in zip(src, tgt):
                f.write(s.strip() + '\t' + t)

"""
test index:
{4, 11, 15, 16, 17, 18, 20, 24, 26, 31, 33, 37, 41, 46, 51, 55, 56, 58, 62, 67, 72, 74, 81, 82, 85, 86, 90, 98, 101, 103, 104, 112, 114, 118, 130, 134, 135, 138, 145, 146, 147, 149, 156, 159, 161, 170, 174, 175, 177, 178, 182, 185, 191, 194, 203, 205, 208, 212, 214, 216, 225, 231, 235, 236, 237, 241, 242, 246, 249, 250, 256, 261, 263, 265, 269, 270, 273, 278, 280, 281, 282, 287, 288, 290, 291, 294, 295, 297, 301, 302, 303, 307, 312, 316, 318, 325, 328, 329, 330, 331, 338, 347, 351, 352, 353, 355, 356, 358, 359, 361, 367, 371, 372, 373, 377, 379, 384, 385, 386, 393, 396, 398, 402, 410, 413, 415, 417, 418, 420, 423, 427, 428, 435, 437, 439, 442, 444, 446, 450, 451, 460, 461, 464, 465, 466, 472, 476, 478, 485, 486, 487, 491, 512, 513, 516, 519, 523, 528, 531, 532, 533, 537, 538, 539, 540, 550, 551, 552, 554, 555, 557, 559, 562, 564, 565, 567, 568, 573, 576, 577, 578, 581, 584, 585, 590, 595, 597, 598, 599, 603, 604, 605, 607, 610, 615, 616, 623, 625, 626, 632, 635, 640, 643, 645, 647, 650, 653, 656, 661, 662, 669, 670, 671, 673, 676, 679, 682, 691, 693, 698, 700, 703, 704, 727, 732, 735, 746, 748, 751, 752, 753, 754, 756, 758, 769, 771, 772, 782, 786, 788, 790, 791, 799, 801, 805, 808, 809, 810, 815, 821, 824, 827, 832, 833, 838, 844, 845, 847, 853, 859, 866, 868, 874, 876, 877, 878, 880, 885, 889, 890, 893, 897, 898, 906, 908, 920, 922, 924, 926, 929, 930, 931, 940, 943, 944, 945, 951, 961, 962, 963, 967, 968, 971, 972, 974, 975, 976, 986, 987, 988, 991, 995, 996, 997, 999, 1000, 1008, 1023, 1026, 1030, 1031, 1037, 1039, 1040, 1042, 1050, 1055, 1059, 1063, 1075, 1077, 1084, 1088, 1093, 1094, 1104, 1105, 1106, 1107, 1109, 1111, 1112, 1113, 1120, 1125, 1128, 1130, 1133, 1135, 1138, 1141, 1144, 1145, 1148, 1149, 1152, 1157, 1158, 1161, 1162, 1166, 1168, 1171, 1173, 1174, 1175, 1179, 1183, 1186, 1187, 1188, 1193, 1195, 1196, 1197, 1204, 1205, 1208, 1216, 1222, 1225, 1226, 1227, 1230, 1234, 1235, 1238, 1241, 1243, 1251, 1253, 1264, 1265, 1268, 1276, 1279, 1281, 1284, 1289, 1292, 1294, 1298, 1300, 1307, 1308, 1311, 1313, 1318, 1320, 1322, 1330, 1332, 1333, 1334, 1337, 1339, 1342, 1345, 1346, 1347, 1351, 1352, 1354, 1358, 1364, 1367, 1368, 1375, 1378, 1379, 1380, 1381, 1386, 1391, 1397, 1400, 1405, 1410, 1412, 1415, 1416, 1418, 1420, 1424, 1426, 1431, 1434, 1435, 1438, 1439, 1441, 1442, 1444, 1445, 1446, 1453, 1455, 1460, 1462, 1463, 1470, 1487, 1490, 1492, 1493, 1496, 1499, 1501, 1502, 1504, 1506, 1507, 1508, 1511, 1514, 1515, 1518, 1519, 1524, 1525, 1526, 1531, 1533, 1534, 1541, 1543, 1546, 1547, 1549, 1550, 1551, 1554, 1555, 1560, 1563, 1566, 1571, 1573, 1576, 1578, 1588, 1595, 1599, 1602, 1606, 1608, 1609, 1611, 1612, 1622, 1624, 1628, 1629, 1640, 1642, 1646, 1654, 1658, 1659, 1660, 1662, 1664, 1666, 1668, 1671, 1672, 1673, 1674, 1677, 1682, 1683, 1684, 1686, 1690, 1693, 1695, 1703, 1713, 1714, 1715, 1720, 1722, 1727, 1732, 1734, 1741, 1751, 1753, 1754, 1758, 1760, 1762, 1763, 1765, 1771, 1773, 1777, 1783, 1789, 1793, 1794, 1798, 1806, 1809, 1810, 1812, 1816, 1822, 1828, 1832, 1835, 1838, 1839, 1841, 1847, 1848, 1849, 1850, 1853, 1857, 1860, 1866, 1868, 1871, 1873, 1875, 1881, 1883, 1884, 1885, 1889, 1891, 1893, 1897, 1899, 1900, 1901, 1906, 1907, 1908, 1914, 1918, 1920, 1922, 1932, 1933, 1934, 1935, 1947, 1951, 1953, 1957, 1958, 1961, 1964, 1965, 1968, 1977, 1996, 1997, 2002, 2010, 2011, 2012, 2013, 2015, 2019, 2025, 2039, 2046, 2050, 2060, 2061, 2063, 2064, 2074, 2075, 2078, 2081, 2084, 2086, 2088, 2089, 2092, 2093, 2094, 2096, 2098, 2099, 2101, 2103, 2104, 2108, 2109, 2110, 2115, 2117, 2119, 2125, 2135, 2136, 2137, 2138, 2141, 2146, 2148, 2149, 2152, 2154, 2156, 2157, 2168, 2171, 2182, 2184, 2191, 2195, 2203, 2206, 2208, 2209, 2211, 2214, 2215, 2217, 2218, 2220, 2228, 2229, 2235, 2245, 2247, 2249, 2253, 2254, 2256, 2259, 2264, 2267, 2270, 2271, 2273, 2274, 2275, 2276, 2285, 2289, 2291, 2294, 2300, 2302, 2313, 2314, 2322, 2326, 2327, 2330, 2331, 2333, 2335, 2336, 2339, 2342, 2345, 2349, 2350, 2352, 2355, 2358, 2360, 2368, 2369, 2370, 2378, 2380, 2381, 2383, 2385, 2386, 2387, 2390, 2399, 2403, 2407, 2413, 2421, 2422, 2423, 2425, 2438, 2443, 2444, 2445, 2448, 2450, 2460, 2477, 2478, 2479, 2480, 2483, 2488, 2489, 2495, 2497, 2501, 2505, 2514, 2516, 2518, 2523, 2527, 2531, 2533, 2536, 2542, 2543, 2549, 2550, 2565, 2567, 2572, 2573, 2574, 2576, 2577, 2587, 2591, 2592, 2596, 2597, 2600, 2603, 2606, 2608, 2609, 2610, 2612, 2613, 2614, 2615, 2619, 2629, 2637, 2640, 2645, 2650, 2655, 2659, 2665, 2674, 2678, 2681, 2682, 2685, 2689, 2696, 2697, 2712, 2717, 2718, 2719, 2723, 2724, 2727, 2729, 2730, 2732, 2733, 2736, 2737, 2738, 2742, 2743, 2744, 2745, 2747, 2748, 2750, 2754, 2755, 2757, 2758, 2761, 2763, 2765, 2766, 2767, 2770, 2780, 2781, 2784, 2785, 2787, 2790, 2796, 2797, 2798, 2799, 2803, 2804, 2813, 2816, 2817, 2821, 2822, 2823, 2830, 2832, 2833, 2839, 2841, 2844, 2845, 2848, 2849, 2854, 2856, 2863, 2865, 2866, 2867, 2877, 2883, 2885, 2888, 2899, 2902, 2908, 2921, 2922, 2924, 2927, 2928, 2932, 2934, 2941, 2946, 2948, 2949, 2954, 2960, 2968, 2971, 2974, 2981, 2982, 2985, 2987, 2988, 2991, 2994, 2995, 2999, 3001, 3004, 3015, 3016, 3019, 3025, 3028, 3031, 3033, 3035, 3037, 3038, 3051, 3053, 3056, 3058, 3059, 3062, 3063, 3064, 3066, 3070, 3072, 3073, 3074, 3081, 3087, 3092, 3096, 3100, 3101, 3103, 3107, 3112, 3114, 3124, 3131, 3135, 3137, 3138, 3140, 3141, 3143, 3145, 3147, 3150, 3151, 3152, 3153, 3158, 3161, 3162, 3163, 3164, 3167, 3168, 3171, 3179, 3180, 3182, 3185, 3188, 3190, 3191, 3206, 3210, 3211, 3220, 3234, 3242, 3243, 3244, 3245, 3250, 3254, 3255, 3260, 3261, 3264, 3268, 3269, 3275, 3279, 3280, 3285, 3300, 3302, 3309, 3316, 3322, 3325, 3326, 3332, 3333, 3339, 3342, 3349, 3355, 3364, 3368, 3372, 3373, 3382, 3385, 3390, 3397, 3399, 3402, 3408, 3414, 3415, 3418, 3422, 3426, 3428, 3429, 3430, 3433, 3440, 3442, 3443, 3452, 3455, 3456, 3457, 3461, 3464, 3466, 3473, 3480, 3490, 3498, 3503, 3509, 3510, 3514, 3515, 3525, 3528, 3529, 3533, 3538, 3542, 3546, 3547, 3550, 3553, 3557, 3563, 3565, 3577, 3578, 3580, 3581, 3582, 3585, 3587, 3590, 3598, 3599, 3606, 3607, 3608, 3609, 3610, 3615, 3626, 3627, 3628, 3630, 3632, 3634, 3637, 3642, 3643, 3651, 3653, 3655, 3657, 3659, 3660, 3663, 3666, 3667, 3668, 3670, 3673, 3676, 3679, 3686, 3687, 3692, 3693, 3696, 3702, 3703, 3706, 3707, 3715, 3717, 3726, 3727, 3728, 3732, 3733, 3734, 3735, 3736, 3740, 3743, 3745, 3749, 3752, 3753, 3754, 3755, 3757, 3758, 3759, 3761, 3762, 3764, 3765, 3768, 3788, 3791, 3795, 3800, 3801, 3802, 3803, 3804, 3808, 3809, 3811, 3812, 3813, 3814, 3816, 3817, 3818, 3820, 3821, 3831, 3832, 3833, 3840, 3841, 3849, 3853, 3856, 3857, 3859, 3860, 3861, 3864, 3866, 3871, 3886, 3888, 3889, 3891, 3894, 3896, 3900, 3901, 3903, 3906, 3912, 3913, 3914, 3920, 3926, 3927, 3929, 3931, 3933, 3936, 3939, 3940, 3949, 3950, 3953, 3954, 3955, 3960, 3964, 3966, 3971, 3974, 3975, 3979, 3982, 3984, 3989, 3993, 3996, 3997, 3999, 4000, 4010, 4012, 4014, 4015, 4018, 4022, 4026, 4027, 4028, 4030, 4031, 4035, 4042, 4045, 4046, 4056, 4057, 4060, 4066, 4067, 4070, 4071, 4076, 4078, 4079, 4083, 4085, 4087, 4098, 4108, 4113, 4117, 4126, 4132, 4135, 4137, 4139, 4140, 4144, 4153, 4158, 4160, 4166, 4167, 4168, 4170, 4172, 4175, 4177, 4179, 4185, 4187, 4195, 4196, 4198, 4202, 4208, 4213, 4216, 4219, 4220, 4222, 4226, 4231, 4234, 4239, 4242, 4244, 4246, 4247, 4250, 4252, 4253, 4258, 4259, 4261, 4263, 4265, 4268, 4275, 4281, 4282, 4285, 4291, 4292, 4298, 4300, 4303, 4310, 4311, 4314, 4321, 4323, 4324, 4327, 4335, 4342, 4343, 4345, 4355, 4359, 4363, 4370, 4380, 4384, 4386, 4388, 4391, 4394, 4398, 4399, 4403, 4409, 4412, 4418, 4419, 4421, 4422, 4427, 4430, 4432, 4435, 4436, 4440, 4443, 4445, 4446, 4450, 4451, 4457, 4458, 4463, 4468, 4472, 4475, 4477, 4481, 4486, 4488, 4490, 4505, 4508, 4522, 4523, 4526, 4531, 4538, 4547, 4551, 4553, 4557, 4562, 4563, 4565, 4568, 4571, 4580, 4582, 4583, 4584, 4585, 4587, 4589, 4590, 4591, 4595, 4598, 4599, 4601, 4603, 4612, 4613, 4619, 4621, 4632, 4633, 4640, 4645, 4647, 4653, 4656, 4657, 4662, 4663, 4665, 4666, 4671, 4673, 4682, 4687, 4688, 4690, 4692, 4699, 4702, 4705, 4714, 4721, 4727, 4731, 4732, 4733, 4734, 4735, 4741, 4745, 4752, 4755, 4764, 4765, 4766, 4768, 4769, 4770, 4774, 4776, 4778, 4779, 4780, 4783, 4787, 4793, 4798, 4802, 4804, 4806, 4807, 4813, 4814, 4815, 4818, 4825, 4829, 4830, 4832, 4834, 4835, 4843, 4844, 4846, 4850, 4852, 4856, 4861, 4864, 4865, 4869, 4870, 4875, 4881, 4886, 4890, 4891, 4894, 4899, 4900, 4904, 4907, 4909, 4910, 4913, 4915, 4920, 4921, 4926, 4927, 4930, 4933, 4936, 4937, 4943, 4944, 4946, 4947, 4948, 4949, 4950, 4953, 4958, 4961, 4962, 4968, 4969, 4970, 4974, 4976, 4982, 4989, 4994, 5006, 5008, 5010, 5014, 5015, 5017, 5022, 5025, 5031, 5034, 5039, 5043, 5047, 5052, 5053, 5056, 5058, 5064, 5065, 5066, 5069, 5072, 5075, 5081, 5082, 5085, 5086, 5089, 5099, 5100, 5102, 5103, 5111, 5114, 5115, 5117, 5119, 5122, 5126, 5128, 5131, 5134, 5139, 5140, 5142, 5145, 5148, 5153, 5155, 5158, 5159, 5161, 5162, 5168, 5177, 5180, 5181, 5186, 5187, 5195, 5200, 5203, 5211, 5212, 5214, 5215, 5216, 5217, 5219, 5220, 5221, 5225, 5226, 5229, 5231, 5233, 5234, 5237, 5238, 5241, 5244, 5245, 5275, 5291, 5293, 5295, 5296, 5297, 5299, 5301, 5305, 5306, 5311, 5312, 5315, 5316, 5318, 5321, 5323, 5324, 5329, 5330, 5334, 5335, 5337, 5338, 5342, 5346, 5347, 5351, 5353, 5359, 5368, 5370, 5373, 5376, 5378, 5383, 5384, 5387, 5394, 5396, 5398, 5400, 5401, 5402, 5404, 5408, 5409, 5414, 5419, 5430, 5438, 5443, 5447, 5448, 5450, 5451, 5452, 5454, 5455, 5460, 5467, 5470, 5472, 5475, 5477, 5480, 5488, 5490, 5492, 5496, 5500, 5502, 5513, 5519, 5525, 5526, 5529, 5531, 5532, 5535, 5552, 5553, 5554, 5564, 5567, 5571, 5572, 5574, 5583, 5584, 5590, 5591, 5594, 5595, 5596, 5597, 5600, 5602, 5606, 5609, 5612, 5627, 5628, 5631, 5635, 5638, 5641, 5652, 5656, 5657, 5659, 5663, 5668, 5670, 5671, 5672, 5674, 5683, 5690, 5692, 5699, 5704, 5710, 5714, 5715, 5718, 5722, 5725, 5726, 5728, 5729, 5739, 5742, 5745, 5747, 5751, 5753, 5757, 5758, 5760, 5762, 5763, 5765, 5767, 5768, 5774, 5776, 5780, 5781, 5784, 5785, 5788, 5794, 5802, 5803, 5810, 5811, 5824, 5825, 5828, 5829, 5842, 5843, 5844, 5846, 5852, 5857, 5858, 5864, 5865, 5866, 5875, 5877, 5879, 5882, 5885, 5886, 5889, 5890, 5892, 5899, 5909, 5910, 5914, 5916, 5917, 5921, 5922, 5923, 5930, 5935, 5936, 5949, 5956, 5961, 5962, 5963, 5965, 5966, 5967, 5971, 5972, 5973, 5974, 5976, 5981, 5983, 5984, 5985, 5987, 5988, 5989, 5997, 5998, 6000, 6009, 6010, 6015, 6016, 6017, 6021, 6023, 6024, 6031, 6035, 6036, 6041, 6044, 6046, 6048, 6051, 6052, 6057, 6062, 6065, 6066, 6067, 6072, 6073, 6074, 6084, 6088, 6090, 6102, 6103, 6105, 6107, 6108, 6109, 6121, 6127, 6129, 6131, 6132, 6134, 6136, 6137, 6139, 6140, 6141, 6143, 6145, 6147, 6153, 6166, 6167, 6168, 6169, 6170, 6173, 6174, 6177, 6179, 6183, 6184, 6185, 6188, 6190, 6192, 6196, 6197, 6203, 6204, 6206, 6215, 6219, 6221, 6222, 6223, 6237, 6244, 6245, 6246, 6247, 6252, 6253, 6254, 6255, 6256, 6259, 6267, 6270, 6272, 6280, 6281, 6286, 6288, 6295, 6299, 6303, 6305, 6307, 6309, 6310, 6313, 6315, 6317, 6323, 6324, 6326, 6329, 6333, 6335, 6341, 6342, 6352, 6356, 6361, 6363, 6365, 6366, 6369, 6370, 6371, 6374, 6379, 6386, 6387, 6388, 6393, 6398, 6400, 6402, 6407, 6408, 6410, 6412, 6413, 6415, 6428, 6429, 6433, 6448, 6449, 6453, 6458, 6460, 6468, 6469, 6474, 6476, 6479, 6481, 6485, 6488, 6491, 6495, 6497, 6498, 6508, 6514, 6515, 6518, 6519, 6521, 6524, 6526, 6527, 6528, 6533, 6535, 6536, 6539, 6541, 6546, 6547, 6551, 6554, 6557, 6562, 6564, 6565, 6566, 6568, 6571, 6578, 6580, 6582, 6585, 6586, 6587, 6588, 6590, 6591, 6594, 6595, 6598, 6601, 6613, 6614, 6616, 6621, 6626, 6629, 6631, 6635, 6636, 6641, 6642, 6651, 6652, 6663, 6670, 6674, 6675}
"""
import random

def get_token_sentence_mapping(tokens, exclude_indexs=set()):
    src_filename = "/home/wangdq/data/all.en"
    tgt_filename = "/home/wangdq/data/all.zh"
    tokens_sentence_mapping = {t: [] for t in tokens}
    sentence_token_mapping = {}
    sentences = {}
    with open(src_filename) as f_src, open(tgt_filename) as f_tgt:
        for index, (src, tgt) in enumerate(zip(f_src, f_tgt)):
            if index in exclude_indexs:
                continue
            src_tokens = set(src.strip().split(' '))
            sentences[index] = (src, tgt)
            sentence_token_mapping[index] = []
            for t in tokens:
                if t in src_tokens:
                    tokens_sentence_mapping[t].append(index)
                    sentence_token_mapping[index].append(t)

    return tokens_sentence_mapping, sentence_token_mapping, sentences




def write_file(dirname, hint, _set, tokens, sentences, sentence_token_mapping):
    txt_filename = os.path.join(dirname, hint + '.txt')
    src_filename = os.path.join(dirname, hint + '.en')
    tgt_filename = os.path.join(dirname, hint + '.zh')

    temp_content = set()
    with open(txt_filename, 'w') as f_txt, open(src_filename, 'w') as f_src, open(tgt_filename, 'w') as f_tgt:
        for t in tokens:
            for index in _set[t]:
                if index in temp_content:
                    continue
                temp_content.add(index)
                token_set = '\t'.join(sentence_token_mapping[index])

                f_txt.write(token_set + '\t' + sentences[index][0].strip() + '\t' + sentences[index][1].strip() + '\n')
                f_src.write(sentences[index][0].strip() + '\n')
                f_tgt.write(sentences[index][1].strip() + '\n')


def filter_tokens():
    tokens = get_tokens()
    tokens_sentence_mapping, sentence_token_mapping, sentences = get_token_sentence_mapping(tokens)
    new_tokens = []
    for token, sentences in tokens_sentence_mapping.items():
        if len(sentences) >= 30:
            new_tokens.append(token)
    tokens_filename = "/home/wangdq/data/tokens.txt"
    with open(tokens_filename, 'w') as f:
        for token in new_tokens:
            f.write(token + '\n')





def split_dataset(test_num=10, train_num='all'):
    tokens = get_tokens()
    tokens_sentence_mapping, sentence_token_mapping, sentences = get_token_sentence_mapping(tokens)

    test_mapping = {}
    test_sentences = set()
    for t in tokens:
        num, i = 0, 0
        test_mapping[t] = []
        random.shuffle(tokens_sentence_mapping[t])
        for i in range(len(tokens_sentence_mapping[t])):
            if num >= test_num:
                break
            index = tokens_sentence_mapping[t][i]
            if len(sentence_token_mapping[index]) == 1:
                test_mapping[t].append(index)
                num += 1
                test_sentences.add(index)
        if num != test_num:
            print("error!")
            assert True, "???"


    dirname = "/home/wangdq/data/dataset"

    import pickle
    pickle.dump(test_sentences, open(os.path.join(dirname, 'test.index'), 'wb'))

    write_file(dirname, "test", test_mapping, tokens, sentences, sentence_token_mapping)

    train_mapping = {}
    for t, index_list in tokens_sentence_mapping.items():
        new_index_list = [index for index in index_list if index not in test_sentences]
        new_index_list.sort()
        train_mapping[t] = new_index_list
    write_file(dirname, "train", train_mapping, tokens, sentences, sentence_token_mapping)



def construct_train(train_num=20):
    dirname = "/home/wangdq/data/dataset"
    import pickle
    test_index = pickle.load(open(os.path.join(dirname, 'test.index'), 'rb'))

    tokens = get_tokens()
    tokens_sentence_mapping, sentence_token_mapping, sentences = get_token_sentence_mapping(tokens, test_index)


    train_mapping = {}
    for t, index_list in tokens_sentence_mapping.items():
        if train_num != 'all':
            if len(index_list) < int(train_num):
                print("???")
            index_list = random.sample(index_list, int(train_num))
        index_list.sort()
        train_mapping[t] = index_list

    dirname = os.path.join(dirname, str(train_num))
    os.mkdir(dirname)
    write_file(dirname, "train", train_mapping, tokens, sentences, sentence_token_mapping)


# filter_tokens()

# split_dataset()

construct_train(train_num=5)
construct_train(train_num=10)
construct_train(train_num=15)
construct_train(train_num=20)
construct_train(train_num='all')


"""

/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe train.bpe.en train.en /home/wangdq/cwmt/codes.en
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe train.bpe.zh train.zh /home/wangdq/cwmt/codes.zh
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe test.bpe.en test.en /home/wangdq/cwmt/codes.en
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe test.bpe.zh test.zh /home/wangdq/cwmt/codes.zh

fairseq-preprocess --source-lang en --target-lang zh \
    --trainpref train.bpe  --testpref test.bpe \
    --srcdict /home/data_ti5_c/wangdq/data/few_shot/pretrain/data-bin/dict.en.txt \
    --tgtdict /home/data_ti5_c/wangdq/data/few_shot/pretrain/data-bin/dict.zh.txt \
    --destdir data-bin \
    --workers 10
    
    
fairseq-preprocess --source-lang en --target-lang zh \
    --trainpref train.bpe  --testpref newstest2017.bpe --validpref newsdev2017.bpe \
    --srcdict /home/data_ti5_c/wangdq/data/few_shot/pretrain/data-bin/dict.en.txt \
    --tgtdict /home/data_ti5_c/wangdq/data/few_shot/pretrain/data-bin/dict.zh.txt \
    --destdir data-bin \
    --workers 10
 """