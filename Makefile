.PHONY:	install_appuser \
	install \
	install_docker \
	clean_docker \
	docker_image \
	docker_image_rebuild \
	docker_devrun \
	docker_localrun \
	help

GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
WHITE  := $(shell tput -Txterm setaf 7)
CYAN   := $(shell tput -Txterm setaf 6)
RESET  := $(shell tput -Txterm sgr0)

help: ## Show this help.
	@echo 'Usage:'
	@echo '  ${CYAN}make${RESET} ${GREEN}<target>${RESET}'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} { \
		if (/^[a-zA-Z_-]+:.*?##.*$$/) {printf "  ${GREEN}%-21s${WHITE}%s${RESET}\n", $$1, $$2} \
		else if (/^## .*$$/) {printf "  ${CYAN}%s${RESET}\n", substr($$1,4)} \
		}' $(MAKEFILE_LIST)

install_appuser: 
	adduser --disabled-password appuser
	gpasswd -a appuser sudo
	printf "%s\\n" "appuser ALL=NOPASSWD: ALL" >> /etc/sudoers

install_docker: install_appuser
	mkdir --parents ./tmp/
	chown --recursive appuser: /opt/tigramite/tmp

install: ## Install tigramite.
	sudo python3 ./setup.py install

docker_image: ## Create docker image.
	docker-compose --project-name tigramite_image --file docker/docker-compose-builder.yml build --progress plain $(FLAG)

docker_image_rebuild: ## Re-create docker image, ignoring the cache.
	docker-compose --project-name tigramite_image --file docker/docker-compose-builder.yml build --progress plain --no-cache $(FLAG)

docker_devrun: ## Run dev container.
	docker-compose --project-name tigramite_dev --file docker/docker-compose-development.yml run --rm app || true

