from zensvi.cv import ClassifierPerceptionViT


for perception in ["safer", "livelier", "wealthier", "more beautiful", "more boring", "more depressing"]:
    classifier = ClassifierPerceptionViT(
        perception_study=perception
    )
    dir_input = "downloaded_google_review_photos"
    dir_summary_output = f"perception_output/{perception}"
    classifier.classify(
        dir_input,
        dir_summary_output=dir_summary_output
    )