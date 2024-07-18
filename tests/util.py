def check_bitblas_available():
    result = False
    try:
        import bitblas  # noqa: E402
        result = True
    except ModuleNotFoundError:
        print(" [WARNING] bitblas not installed, Please install via `pip install bitblas`.")
    except Exception as e:
        print(f" [WARNING] Could not load BitBLASQuantLinear: {e}")
    finally:
        return result