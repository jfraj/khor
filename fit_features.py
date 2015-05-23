# List of all possible contries
# from the command
# (nan not included)
FULL_COUNTRY_LIST =\
    ['us', 'in', 'py', 'ru', 'th', 'id', 'za', 'ng', 'sd', 'au', 'hr',
     'np', 'iq', 'bd', 'tr', 'ch', 'ke', 'uk', 'fr', 'pk', 'my', 'vn',
     'ro', 'gh', 'ua', 'pl', 'by', 'ar', 'zm', 'lk', 'ph', 'br', 'es',
     'mx', 'il', 'qa', 'nl', 've', 'sg', 'gt', 'ae', 'az', 'uz', 'ht',
     'tz', 'gm', 'dk', 'no', 'kw', 'mk', 'hu', 'it', 'ml', 'sv', 'bn',
     'ni', 'cn', 'et', 'ge', 'mw', 'ee', 'ye', 'kr', 'tn', 'gr', 'at',
     'cm', 'ca', 'mn', 'rs', 'sz', 'pe', 'jp', 'sl', 'bh', 'zw', 'bg',
     'de', 'eu', 'cr', 'jo', 'ie', 'sa', 'eg', 'dz', 'hk', 'ec', 'si',
     'lv', 'na', 'mt', 'ug', 'kg', 'se', 'bb', 'sc', 'sn', 'om',
     'fi', 'cl', 'ma', 'am', 'lr', 'be', 'bf', 'kh', 'md', 'ly', 'al',
     'ba', 'bo', 'lt', 'ga', 'mr', 'jm', 'bj', 'mu', 'pa', 'cz', 'ao',
     'lu', 'me', 'af', 'kz', 'hn', 'ls', 'uy', 'lb', 'cy', 'sk', 'ir',
     'la', 'dj', 'bz', 'ci', 'is', 'mg', 'so', 'co', 'pt', 'gy', 'td',
     'rw', 'pr', 'bw', 'gq', 'cv', 'mc', 'ne', 'tg', 'bi', 'sy', 'tt',
     'cd', 'sb', 'mz', 'mm', 'tj', 'tw', 'gu', 'cg', 'gl', 'nz', 'mv',
     'ps', 'tm', 'ag', 'ad', 'sr']

MERCHANDISE_LIST = ['jewelry', 'furniture', 'home goods', 'mobile',
                    'sporting goods', 'office equipment', 'computers',
                    'books and music', 'clothing', 'auto parts']


def get_ctry_full_feature_list():
    """Return a list of all the country features."""
    return ["ctry_{}".format(x) for x in FULL_COUNTRY_LIST]


def get_merchandise_rename_dict(**kwargs):
    """return a dictionary mapping csv merchandise name to feature name

    The feature name is intended to be more friendly
    Format:
    {merchandise: feature_name}

    if kwarg inverted = True the dictionary is inverted:
    {feature_name : merchandise}
    """
    inverted = kwargs.get('inverted', False)

    feat_dict = {}
    merch_dict = {}
    for imerch in MERCHANDISE_LIST:
        ifeat_name = "mer_{}".format(imerch[:4])
        assert(ifeat_name not in feat_dict.values())
        feat_dict[imerch] = ifeat_name
        merch_dict[ifeat_name] = imerch
    if inverted:
        return merch_dict
    return feat_dict


def get_merch_full_feature_list():
    """Return list of all merchandise features."""
    return get_merchandise_rename_dict().values()

if __name__ == "__main__":
    print(get_merchandise_rename_dict())
    print(get_merchandise_rename_dict(inverted=True))
    print(get_merch_full_feature_list())
