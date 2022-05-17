if __name__ == '__main__':
    from datetime import datetime
    import nhanes_cvr.app as app

    start = datetime.now()
    # Takes an hour to run all of them with 3 different folding

    app.runHandPickedFeatures()
    print(f"runHandPicked - {datetime.now() - start}")

    app.runHandPickedNoNulls()
    print(f"runHandPickedNoNulls - {datetime.now() - start}")

    app.runCorrelationFeatureSelection()
    print(f"runCorrelationFeatureSelection - {datetime.now() - start}")

    app.runCorrelationFeatureSelectionDropNulls()
    print(
        f"runCorrelationFeatureSelectionDropNulls - {datetime.now() - start}")
