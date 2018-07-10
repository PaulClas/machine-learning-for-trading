"""
inspired by https://github.com/jeremyjordan/stock-price-forecasting
"""

import scrapy
import json
import re
from furl import furl
from scrapy.http import Request
from dateutil.parser import parse as parse_date
from earnings_calls.items import TranscriptItem


test_url = 'https://seekingalpha.com/symbol/GOOG/earnings/more_transcripts?page=1'


class TranscriptSpider(scrapy.Spider):
    name = 'transcripts'
    rotate_user_agent = True

    # Allow the companies for scraping to be specified by ticker symbol
    def __init__(self, symbols=None, *args, **kwargs):
        '''
        Accepts either a single company ticker symbol, or a comma-separated list of company ticker symbols for scraping.
        '''
        super().__init__(*args, **kwargs)
        base_url = 'https://seekingalpha.com/symbol/{}/earnings/more_transcripts?page=1'
        self.start_urls = [base_url.format(s) for s in symbols.split(',')]
        self.logger.info('Start URLs: {}'.format(','.join(self.start_urls)))
        # self.logger.info('UA: {}'.format(self.http.Request)

    def parse(self, response):
        data = json.loads(response.body.decode('utf8'))
        self.logger.info('DATA: ', data)
        self.logger.info(response.request.headers['User-Agent'])
        selector = scrapy.Selector(text=data['html'], type="html")

        if data['count'] > 0:
            # This is necessary when start_urls contains a list of companies
            symbol = re.search(r'(?<=symbol\/)(.*)(?=\/earnings)', response.url).group(0)

            # Select parent node for only the child nodes that contain an earnings call transcript
            contains_call = 'a[contains(text(), "Earnings Call Transcript")]'
            links = [l for l in selector.css('.symbol_article') if l.xpath(contains_call)]

            for link in links:
                transcript = TranscriptItem()

                # Scrape basic info from link
                transcript['title'] = link.xpath(contains_call + '/text()').extract_first()
                url = furl(response.urljoin(link.xpath(contains_call + '/@href').extract_first()))
                transcript['url'] = url.add({'part': 'single'}).url
                transcript['date'] = parse_date(link.css('.date_on_by::text').extract_first())
                transcript['symbol'] = symbol

                # Request transcript url for further scraping, passing what we've collected so far as meta information
                request = Request(transcript['url'], callback=self.parse_transcript)
                request.meta['transcript'] = transcript
                yield request

            f = furl(response.url)
            f.args['page'] = str(int(f.args['page']) + 1)
            next_page = str(f.url)
            yield scrapy.Request(next_page)

    def parse_transcript(self, response):
        transcript = response.meta['transcript']

        scraped = [''.join(x.xpath('./descendant::text()').extract()) for x in
                   response.xpath('//div[@id="a-body"]/p')]
        self.logger.info(scraped)

        try:
            # Split transcript into list of executives on the call, list of analysts on the call, and the actual transcript
            # Different transcripts have iterations of the following terms, so can't use .index("Executives")
            # ie. Some transcripts have "Executives" while others have "Executives: "
            executives_start = [i for i, j in enumerate(scraped) if "Executive" in j][0]
            analysts_start = [i for i, j in enumerate(scraped) if "Analyst" in j][0]
            operator_start = [i for i, j in enumerate(scraped) if "Operator" in j][0]
            copyright_start = transcript.index("Copyright policy:")

            transcript['executives'] = transcript[executives_start + 1:analysts_start]
            transcript['analysts'] = transcript[analysts_start + 1:operator_start]
            transcript['body'] = transcript[operator_start + 1:copyright_start]
            self.logger.info(transcript)
        except:
            # If the formatting is really strange, just store everything in body and flag the anomoly in executives/analyst fields
            transcript['executives'] = 'Parsing error'
            transcript['analysts'] = 'Parsing error'
            transcript['body'] = transcript
            self.logger.info(transcript)

        yield transcript
