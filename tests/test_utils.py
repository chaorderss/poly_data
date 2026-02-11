import pytest
import os
import tempfile
import csv
import polars as pl
from poly_utils.utils import get_markets, update_missing_tokens, PLATFORM_WALLETS


class TestGetMarkets:
    """Test cases for get_markets function"""

    def test_get_markets_with_no_files(self, tmp_path):
        """Test get_markets when no market files exist"""
        main_file = str(tmp_path / "nonexistent_markets.csv")
        missing_file = str(tmp_path / "nonexistent_missing.csv")
        
        result = get_markets(main_file, missing_file)
        
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0

    def test_get_markets_with_main_file_only(self, tmp_path):
        """Test get_markets with only main markets file"""
        main_file = str(tmp_path / "markets.csv")
        missing_file = str(tmp_path / "missing_markets.csv")
        
        # Create sample main markets file
        with open(main_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['createdAt', 'id', 'question', 'answer1', 'answer2', 'neg_risk', 
                           'market_slug', 'token1', 'token2', 'condition_id', 'volume', 'ticker', 'closedTime'])
            writer.writerow(['2024-01-01', 'market1', 'Test Question 1?', 'Yes', 'No', False,
                           'test-slug-1', 'token1', 'token2', 'cond1', '1000', 'TEST1', '2024-12-31'])
            writer.writerow(['2024-01-02', 'market2', 'Test Question 2?', 'Yes', 'No', False,
                           'test-slug-2', 'token3', 'token4', 'cond2', '2000', 'TEST2', '2024-12-31'])
        
        result = get_markets(main_file, missing_file)
        
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2
        assert 'id' in result.columns
        assert result['id'][0] == 'market1'
        assert result['id'][1] == 'market2'

    def test_get_markets_deduplicates(self, tmp_path):
        """Test that get_markets deduplicates markets with same ID"""
        main_file = str(tmp_path / "markets.csv")
        missing_file = str(tmp_path / "missing_markets.csv")
        
        # Create main file with duplicate
        with open(main_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['createdAt', 'id', 'question', 'answer1', 'answer2', 'neg_risk', 
                           'market_slug', 'token1', 'token2', 'condition_id', 'volume', 'ticker', 'closedTime'])
            writer.writerow(['2024-01-01', 'market1', 'Test Question 1?', 'Yes', 'No', False,
                           'test-slug-1', 'token1', 'token2', 'cond1', '1000', 'TEST1', '2024-12-31'])
        
        # Create missing file with same market ID
        with open(missing_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['createdAt', 'id', 'question', 'answer1', 'answer2', 'neg_risk', 
                           'market_slug', 'token1', 'token2', 'condition_id', 'volume', 'ticker', 'closedTime'])
            writer.writerow(['2024-01-01', 'market1', 'Test Question 1 Updated?', 'Yes', 'No', False,
                           'test-slug-1-updated', 'token1', 'token2', 'cond1', '1500', 'TEST1', '2024-12-31'])
        
        result = get_markets(main_file, missing_file)
        
        assert len(result) == 1  # Should only have one market after deduplication
        assert result['id'][0] == 'market1'

    def test_get_markets_sorts_by_createdAt(self, tmp_path):
        """Test that get_markets sorts markets by createdAt"""
        main_file = str(tmp_path / "markets.csv")
        missing_file = str(tmp_path / "missing_markets.csv")
        
        # Create markets with different creation dates (unsorted)
        with open(main_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['createdAt', 'id', 'question', 'answer1', 'answer2', 'neg_risk', 
                           'market_slug', 'token1', 'token2', 'condition_id', 'volume', 'ticker', 'closedTime'])
            writer.writerow(['2024-01-03', 'market3', 'Test Question 3?', 'Yes', 'No', False,
                           'test-slug-3', 'token5', 'token6', 'cond3', '3000', 'TEST3', '2024-12-31'])
            writer.writerow(['2024-01-01', 'market1', 'Test Question 1?', 'Yes', 'No', False,
                           'test-slug-1', 'token1', 'token2', 'cond1', '1000', 'TEST1', '2024-12-31'])
            writer.writerow(['2024-01-02', 'market2', 'Test Question 2?', 'Yes', 'No', False,
                           'test-slug-2', 'token3', 'token4', 'cond2', '2000', 'TEST2', '2024-12-31'])
        
        result = get_markets(main_file, missing_file)
        
        assert len(result) == 3
        assert result['id'][0] == 'market1'  # Earliest date
        assert result['id'][1] == 'market2'  # Middle date
        assert result['id'][2] == 'market3'  # Latest date


class TestUpdateMissingTokens:
    """Test cases for update_missing_tokens function"""

    def test_update_missing_tokens_empty_list(self, tmp_path):
        """Test update_missing_tokens with empty token list"""
        csv_file = str(tmp_path / "missing_markets.csv")
        
        update_missing_tokens([], csv_file)
        
        # Should not create file for empty list
        assert not os.path.exists(csv_file)

    def test_update_missing_tokens_creates_file(self, tmp_path, monkeypatch):
        """Test that update_missing_tokens creates file with correct structure"""
        csv_file = str(tmp_path / "missing_markets.csv")
        
        # Mock the requests.get to avoid actual API calls
        class MockResponse:
            status_code = 200
            def json(self):
                return [{
                    'id': 'test_market_1',
                    'createdAt': '2024-01-01',
                    'question': 'Test Question?',
                    'outcomes': ['Yes', 'No'],
                    'clobTokenIds': ['token1', 'token2'],
                    'negRiskAugmented': False,
                    'negRiskOther': False,
                    'slug': 'test-slug',
                    'conditionId': 'cond1',
                    'volume': '1000',
                    'events': [{'ticker': 'TEST'}],
                    'closedTime': '2024-12-31'
                }]
        
        import requests
        def mock_get(*args, **kwargs):
            return MockResponse()
        
        monkeypatch.setattr(requests, "get", mock_get)
        
        update_missing_tokens(['token1'], csv_file)
        
        assert os.path.exists(csv_file)
        
        # Verify file has correct headers
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert 'id' in headers
            assert 'question' in headers
            assert 'token1' in headers
            assert 'token2' in headers


class TestPlatformWallets:
    """Test that PLATFORM_WALLETS constant is defined"""

    def test_platform_wallets_exists(self):
        """Test that PLATFORM_WALLETS is defined and is a list"""
        assert isinstance(PLATFORM_WALLETS, list)
        assert len(PLATFORM_WALLETS) > 0

    def test_platform_wallets_contains_addresses(self):
        """Test that PLATFORM_WALLETS contains valid Ethereum addresses"""
        for wallet in PLATFORM_WALLETS:
            assert isinstance(wallet, str)
            assert wallet.startswith('0x')
            assert len(wallet) == 42  # Ethereum addresses are 42 characters
