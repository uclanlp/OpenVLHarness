from .smp import dump, load, localize_df


def LOCALIZE(fname, new_fname=None):
    if new_fname is None:
        new_fname = fname.replace('.tsv', '_local.tsv')

    dname = fname.split('/')[-1].rsplit('.', 1)[0]
    data = load(fname)
    data_new = localize_df(data, dname)
    dump(data_new, new_fname)
    return new_fname
