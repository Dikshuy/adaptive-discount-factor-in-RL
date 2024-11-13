def load_map(name):

    EASY_SPARSE = [
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
    ]

    EASY_MEDIUM = [
        "00000",
        "00020",
        "00000",
        "00200",
        "00002",
    ]

    EASY_DENSE = [
        "00000",
        "02220",
        "00000",
        "02200",
        "00002",
    ]

    MODERATE_SPARSE = [
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "11011",
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
    ]

    MODERATE_MEDIUM = [
        "00000",
        "00200",
        "00000",
        "02220",
        "00000",
        "11011",
        "02000",
        "00020",
        "00000",
        "00200",
        "00000",
    ]

    MODERATE_DENSE = [
        "00000",
        "02222",
        "00000",
        "02220",
        "00000",
        "11011",
        "02000",
        "00220",
        "00000",
        "00220",
        "00000",
    ]


    DIFFICULT_SPARSE = [
        "00000100000",
        "00000100000",
        "00000100000",
        "00000000000",
        "00000100000",
        "11011111101",
        "00000100000",
        "00000100000",
        "00000100000",
        "00000000000",
        "00000100000",
    ]

    DIFFICULT_MEDIUM = [
        "00000100000",
        "00200100022",
        "00000100000",
        "02220000020",
        "00000100000",
        "11011111101",
        "02000100000",
        "00020102020",
        "00000100000",
        "00200002000",
        "00000100000",
    ]

    DIFFICULT_DENSE = [
        "00000100000",
        "02222100022",
        "00000120002",
        "02220002020",
        "00000100000",
        "11011111101",
        "02000100200",
        "00220102020",
        "00000100000",
        "00220002002",
        "00000100002",
    ]
    
    if name == 'EASY_SPARSE':
        return EASY_SPARSE
    elif name == 'EASY_MEDIUM':
        return EASY_MEDIUM
    elif name == 'EASY_DENSE':
        return EASY_DENSE
    elif name == 'MODERATE_SPARSE':
        return MODERATE_SPARSE
    elif name == 'MODERATE_MEDIUM':
        return MODERATE_MEDIUM
    elif name == 'MODERATE_DENSE':
        return MODERATE_DENSE
    elif name == 'DIFFICULT_SPARSE':
        return DIFFICULT_SPARSE
    elif name == 'DIFFICULT_MEDIUM':
        return DIFFICULT_MEDIUM
    elif name == 'DIFFICULT_DENSE':
        return DIFFICULT_DENSE
    else:
        NotImplementedError("Check environment name!")