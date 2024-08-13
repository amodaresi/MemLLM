MAX_QUERY_RES_LEN = 30
MAX_QUERIES = 3

SKIP_QUERIES = [
    (None, 'country of citizenship', 'X'),
    (None, 'country', 'X'),
    (None, 'country of origin', 'X'),
    (None, 'religion', 'X'),
    (None, 'place of birth', 'X'),
    (None, 'place of death', 'X'),
    (None, 'work location', 'X'),
    (None, 'location of formation', 'X'),
    (None, 'location', 'X'),
    (None, 'basin country', 'X'),
    (None, 'residence', 'X'),
    (None, 'publication date', 'X'),
    (None, 'production company', 'X'),
    ('X', 'continent', None),
    (None, 'original language of work', 'X'),
    (None, 'applies to jurisdiction', 'X'),
    (None, 'platform', 'X'),
    (None, 'located in the administrative territorial entity', 'X'),
    ('X', 'contains administrative territorial entity', None),
    # (None, 'continent', 'X')
] + [
    (None, 'headquarters location', 'X'),
    (None, 'inception', 'X'),
    (None, 'employer', 'X'),
    (None, 'date of birth', 'X'),
    (None, 'date of death', 'X'),
    (None, 'educated at', 'X'),
]