# -*- coding: utf-8 -*-

# Scrapy settings for data project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     http://doc.scrapy.org/en/latest/topics/settings.html
#     http://scrapy.readthedocs.org/en/latest/topics/downloader-middleware.html
#     http://scrapy.readthedocs.org/en/latest/topics/spider-middleware.html

BOT_NAME = 'earnings_transcripts'

SPIDER_MODULES = ['earnings_calls.spiders']

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests performed by Scrapy (default: 16)
CONCURRENT_REQUESTS = 1
DOWNLOAD_DELAY = 3

# Disable cookies (enabled by default)
COOKIES_ENABLED = False

# Enable or disable downloader middlewares
# See http://scrapy.readthedocs.org/en/latest/topics/downloader-middleware.html
# See https://github.com/alecxe/scrapy-fake-useragent
DOWNLOADER_MIDDLEWARES = {
    'earnings_calls.middlewares.RotateUserAgentMiddleware'             : 110,
    'scrapy.downloadermiddleware.retry.RetryMiddleware'        : 90,
    'scrapy.downloadermiddleware.httpproxy.HttpProxyMiddleware': 110,
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware'       : None,
    'scrapy_fake_useragent.middleware.RandomUserAgentMiddleware'       : 400,
    'tutorial.randomproxy.RandomProxy'                                 : 100,
}

USER_AGENT_LIST = "/home/stefan/Dropbox/Packt/Algorithmic Trading/algo_trading/02_data/scraping/earnings_calls/user_agents.txt"

# Enable or disable extensions
# See http://scrapy.readthedocs.org/en/latest/topics/extensions.html

EXTENSIONS = {
    'earnings_calls.extensions.MonitorDownloadsExtension': 100,
    'earnings_calls.extensions.DumpStatsExtension'       : 101,
    'scrapy.extensions.logstats.LogStats'                : 500,
}

# Configure item pipelines
# See http://scrapy.readthedocs.org/en/latest/topics/item-pipeline.html
ITEM_PIPELINES = {
    'earnings_calls.pipelines.DataPipeline': 300,
}

# Enable and configure the AutoThrottle extension (disabled by default)
# See http://doc.scrapy.org/en/latest/topics/autothrottle.html
AUTOTHROTTLE_ENABLED = True
# The initial download delay
AUTOTHROTTLE_START_DELAY = 3
# The maximum download delay to be set in case of high latencies
AUTOTHROTTLE_MAX_DELAY = 30
# The average number of requests Scrapy should be sending in parallel to
# each remote server
# AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
# Enable showing throttling stats for every response received:
# AUTOTHROTTLE_DEBUG = False


FEED_EXPORTERS = {
    'jsonlines': 'scrapy.exporters.JsonLinesItemExporter',
    'csv'      : 'scrapy.exporters.CsvItemExporter',
}
FEED_FORMAT = 'csv'
FEED_URI = "data/transcript.csv"
# FEED_URI = "company_transcripts/transcripts.json"     # use this line for crawling multiple companies

LOG_LEVEL = 'DEBUG'
LOG_FILE = 'spider.log'

MONITOR_DOWNLOADS_INTERVAL = 10
DUMP_STATS_INTERVAL = 60
