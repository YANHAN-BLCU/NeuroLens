"""合成中文 NER 训练数据：姓名(PER) + 详细地址(ADDR)"""
import json
import random
from pathlib import Path

random.seed(42)

SURNAMES = list("王李张刘陈杨赵黄周吴徐孙胡朱高林何郭马罗梁宋郑谢韩唐冯于董萧程曹袁邓许傅沈曾彭吕苏卢蒋蔡贾丁魏薛叶阎余潘杜戴夏钟汪田任姜范方石姚谭廖邹熊金陆郝孔白崔康毛邱秦江史顾侯邵孟龙万段雷钱汤尹黎易常武乔贺赖龚文")
GIVEN2 = ["伟", "芳", "娜", "敏", "静", "丽", "强", "磊", "军", "洋", "勇", "艳", "杰", "娟", "涛", "明", "超", "秀英", "华", "慧", "建国", "建军", "志强", "晓明", "佳怡", "子豪", "雪梅", "文博", "思远", "雨桐", "梓涵", "浩然", "欣怡", "宇轩", "诗涵"]
GIVEN1 = list("伟芳娜敏静丽强磊军洋勇艳杰娟涛明超华慧婷磊刚平辉鹏玲琳鑫蕾琪瑶璐彤睿博洋阳晨航翔凯瑞宁欣妍")

REGIONS = [
    ("北京市", "朝阳区", ["建国路", "朝阳路", "望京街", "三里屯路", "东三环北路"]),
    ("北京市", "海淀区", ["中关村大街", "学院路", "知春路", "西二旗大街", "清华东路"]),
    ("上海市", "浦东新区", ["陆家嘴环路", "世纪大道", "张杨路", "花木路", "金科路"]),
    ("上海市", "徐汇区", ["漕溪北路", "虹桥路", "肇嘉浜路", "天钥桥路", "龙华中路"]),
    ("广东省", "深圳市", ["科技园南路", "深南大道", "华强北路", "南山大道", "高新南一道"]),
    ("广东省", "广州市", ["天河路", "珠江新城花城大道", "体育西路", "中山大道", "黄埔大道西"]),
    ("浙江省", "杭州市", ["文三路", "西湖大道", "庆春路", "天目山路", "江南大道"]),
    ("江苏省", "南京市", ["中山北路", "汉中路", "江东中路", "龙蟠中路", "软件大道"]),
    ("四川省", "成都市", ["人民南路", "天府大道", "红星路", "一环路东三段", "科华北路"]),
    ("湖北省", "武汉市", ["珞喻路", "中南路", "解放大道", "光谷大道", "建设大道"]),
    ("山东省", "济南市", ["经十路", "泺源大街", "文化东路", "工业南路", "二环东路"]),
    ("福建省", "厦门市", ["厦禾路", "湖滨南路", "仙岳路", "嘉禾路", "环岛东路"]),
    ("辽宁省", "沈阳市", ["青年大街", "和平大街", "建设大路", "三好街", "浑南中路"]),
    ("湖南省", "长沙市", ["麓山南路", "五一大道", "芙蓉中路", "韶山南路", "枫林二路"]),
    ("河南省", "郑州市", ["农业路", "花园路", "金水路", "中原中路", "经三路"]),
    ("陕西省", "西安市", ["高新四路", "雁塔路", "长安路", "科技路", "未央路"]),
    ("重庆市", "渝中区", ["解放碑步行街", "邹容路", "民权路", "八一路", "中山三路"]),
    ("天津市", "和平区", ["南京路", "解放北路", "大沽北路", "成都道", "卫国道"]),
    ("安徽省", "合肥市", ["望江西路", "长江中路", "潜山路", "金寨路", "黄山路"]),
    ("云南省", "昆明市", ["翠湖南路", "北京路", "人民中路", "西昌路", "环城南路"]),
]

BUILDINGS = ["大厦", "广场", "写字楼", "商务中心", "科技园", "产业园", "小区", "花园", "公寓", "中心"]
PREFIXES_PER = ["联系人", "收件人", "申请人", "客户", "负责人", "经理", "主管", "专员", "董事长", "工程师"]
PREFIXES_ADDR = ["地址", "办公地点", "收货地址", "详细地址", "户籍地址", "配送至", "门店位于", "公司位于", "寄至", "工作地点"]


def rand_name() -> str:
    s = random.choice(SURNAMES)
    if random.random() < 0.35:
        return s + random.choice(GIVEN2)
    return s + random.choice(GIVEN1)


def rand_address() -> str:
    prov, dist, streets = random.choice(REGIONS)
    street = random.choice(streets)
    num = random.randint(1, 999)
    suffix = random.choice(BUILDINGS)
    # 约 30% 带楼栋号
    if random.random() < 0.3:
        return f"{prov}{dist}{street}{num}号{suffix}{random.randint(1, 20)}栋"
    return f"{prov}{dist}{street}{num}号{suffix}"


def make_sample_per_only() -> dict:
    templates = [
        lambda n: (f"{n}的手机号是{random.randint(130, 199)}{random.randint(10000000, 99999999)}", [(0, len(n), "PER")]),
        lambda n: (f"紧急联系人{n}", [(5, 5 + len(n), "PER")]),
        lambda n: (f"请{n}签字确认", [(1, 1 + len(n), "PER")]),
        lambda n: (f"研发部{n}已完成任务", [(3, 3 + len(n), "PER")]),
        lambda n: (f"{n}", [(0, len(n), "PER")]),
        lambda n: (f"项目经理{n}来电反馈问题", [(4, 4 + len(n), "PER")]),
    ]
    name = rand_name()
    text, spans = random.choice(templates)(name)
    return {"text": text, "entities": [{"start": s, "end": e, "label": l} for s, e, l in spans]}


def make_sample_addr_only() -> dict:
    addr = rand_address()
    templates = [
        lambda a: (f"收货地址：{a}", [(5, 5 + len(a), "ADDR")]),
        lambda a: (f"仓库位于{a}", [(4, 4 + len(a), "ADDR")]),
        lambda a: (f"请于本周五前将材料送至{a}", [(11, 11 + len(a), "ADDR")]),
        lambda a: (f"配送至{a}", [(3, 3 + len(a), "ADDR")]),
        lambda a: (f"新办公室在{a}", [(5, 5 + len(a), "ADDR")]),
        lambda a: (f"总部地址{a}", [(4, 4 + len(a), "ADDR")]),
    ]
    text, spans = random.choice(templates)(addr)
    return {"text": text, "entities": [{"start": s, "end": e, "label": l} for s, e, l in spans]}


def make_sample_both() -> dict:
    name = rand_name()
    addr = rand_address()
    templates = [
        lambda n, a: (f"请寄至{n}，{a}", [(3, 3 + len(n), "PER"), (3 + len(n) + 1, 3 + len(n) + 1 + len(a), "ADDR")]),
        lambda n, a: (f"收件人：{n}，地址{a}", [(4, 4 + len(n), "PER"), (7 + len(n), 7 + len(n) + len(a), "ADDR")]),
        lambda n, a: (f"联系人{n}，办公地点{a}", [(3, 3 + len(n), "PER"), (8 + len(n), 8 + len(n) + len(a), "ADDR")]),
        lambda n, a: (f"{n}将于下周到{a}报到", [(0, len(n), "PER"), (6 + len(n), 6 + len(n) + len(a), "ADDR")]),
        lambda n, a: (f"请将合同寄给{n}，地址：{a}", [(6, 6 + len(n), "PER"), (10 + len(n), 10 + len(n) + len(a), "ADDR")]),
        lambda n, a: (f"{random.choice(PREFIXES_PER)}{n}，{random.choice(PREFIXES_ADDR)}{a}", None),
        lambda n, a: (f"客户{n}反馈，门店在{a}", [(2, 2 + len(n), "PER"), (8 + len(n), 8 + len(n) + len(a), "ADDR")]),
        lambda n, a: (f"包裹收件人{n}，详细地址：{a}", [(5, 5 + len(n), "PER"), (11 + len(n), 11 + len(n) + len(a), "ADDR")]),
    ]
    tpl = random.choice(templates)
    result = tpl(name, addr)
    if result[1] is None:
        text = result[0]
        # 动态查找偏移
        i_name = text.index(name)
        i_addr = text.index(addr, i_name + len(name))
        spans = [(i_name, i_name + len(name), "PER"), (i_addr, i_addr + len(addr), "ADDR")]
    else:
        text, spans = result
    return {"text": text, "entities": [{"start": s, "end": e, "label": l} for s, e, l in spans]}


def make_sample_double_per() -> dict:
    n1, n2 = rand_name(), rand_name()
    while n2 == n1:
        n2 = rand_name()
    addr = rand_address()
    text = f"{n1}与{n2}同住{addr}"
    return {
        "text": text,
        "entities": [
            {"start": 0, "end": len(n1), "label": "PER"},
            {"start": 1 + len(n1), "end": 1 + len(n1) + len(n2), "label": "PER"},
            {"start": 2 + len(n1) + len(n2), "end": 2 + len(n1) + len(n2) + len(addr), "label": "ADDR"},
        ],
    }


def make_sample_negative() -> dict:
    """无实体样本"""
    texts = [
        "请于下周一提交申请材料，无需邮寄。",
        "本次会议讨论产品迭代计划，地点另行通知。",
        "系统将于今晚零点进行维护，请提前保存数据。",
        "如有疑问请拨打客服热线400-800-1234。",
        "培训资料已上传至内部网盘，请自行下载。",
    ]
    return {"text": random.choice(texts), "entities": []}


def generate(n: int) -> list[dict]:
    makers = [
        (make_sample_both, 0.50),
        (make_sample_per_only, 0.18),
        (make_sample_addr_only, 0.18),
        (make_sample_double_per, 0.09),
        (make_sample_negative, 0.05),
    ]
    samples = []
    for _ in range(n):
        r = random.random()
        cum = 0.0
        for fn, w in makers:
            cum += w
            if r <= cum:
                samples.append(fn())
                break
    return samples


def verify(samples: list[dict]) -> None:
    for i, s in enumerate(samples, 1):
        text = s["text"]
        for ent in s["entities"]:
            span = text[ent["start"] : ent["end"]]
            if not span:
                raise ValueError(f"line {i}: bad span {ent} in {text!r}")
            if ent["label"].upper() == "ADDR" and span.startswith("："):
                raise ValueError(f"line {i}: ADDR span must not include colon: {span!r} in {text!r}")


def main():
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)

    train = generate(100)
    valid = generate(15)
    verify(train)
    verify(valid)

    (out_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in train) + "\n",
        encoding="utf-8",
    )
    (out_dir / "valid.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in valid) + "\n",
        encoding="utf-8",
    )
    print(f"train: {len(train)} samples -> {out_dir / 'train.jsonl'}")
    print(f"valid: {len(valid)} samples -> {out_dir / 'valid.jsonl'}")


if __name__ == "__main__":
    main()
