

deploy: incr_version_release
	@read -r -p "WARNING: This will upload a new public release! Press ENTER to proceed, CTRL-C to cancel."
	rm -f dist/*
	python setup.py sdist
	twine upload dist/*

deploy_dev: test2 test3 incr_version_dev
	@read -r -p "WARNING: This will upload a new development release! Press ENTER to proceed, CTRL-C to cancel."
	rm -f dist/*
	python setup.py sdist
	twine upload dist/*

incr_version_dev:
	python -c 'import setup; setup.increment_version_dev()'

incr_version_release:
	python -c 'import setup; setup.increment_version_release()'
