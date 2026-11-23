import pytest
import os
import csv
import tempfile
from update_utils.update_markets import count_csv_lines, update_markets


class TestCountCsvLines:
    """Test cases for count_csv_lines function"""

    def test_count_csv_lines_nonexistent_file(self):
        """Test count_csv_lines with non-existent file"""
        result = count_csv_lines("nonexistent_file.csv")
        assert result == 0

    def test_count_csv_lines_empty_file(self, tmp_path):
        """Test count_csv_lines with empty file"""
        csv_file = tmp_path / "empty.csv"
        csv_file.touch()
        
        result = count_csv_lines(str(csv_file))
        assert result == 0

    def test_count_csv_lines_header_only(self, tmp_path):
        """Test count_csv_lines with file containing only headers"""
        csv_file = tmp_path / "header_only.csv"
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['header1', 'header2', 'header3'])
        
        result = count_csv_lines(str(csv_file))
        assert result == 0

    def test_count_csv_lines_with_data(self, tmp_path):
        """Test count_csv_lines with file containing data rows"""
        csv_file = tmp_path / "with_data.csv"
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['header1', 'header2', 'header3'])
            writer.writerow(['row1col1', 'row1col2', 'row1col3'])
            writer.writerow(['row2col1', 'row2col2', 'row2col3'])
            writer.writerow(['row3col1', 'row3col2', 'row3col3'])
        
        result = count_csv_lines(str(csv_file))
        assert result == 3

    def test_count_csv_lines_with_empty_rows(self, tmp_path):
        """Test count_csv_lines ignores empty rows"""
        csv_file = tmp_path / "with_empty_rows.csv"
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['header1', 'header2', 'header3'])
            writer.writerow(['row1col1', 'row1col2', 'row1col3'])
            writer.writerow([])  # Empty row
            writer.writerow(['row2col1', 'row2col2', 'row2col3'])
        
        result = count_csv_lines(str(csv_file))
        assert result == 2  # Should not count empty row

    def test_count_csv_lines_large_file(self, tmp_path):
        """Test count_csv_lines with larger file"""
        csv_file = tmp_path / "large_file.csv"
        
        num_rows = 1000
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['header1', 'header2', 'header3'])
            for i in range(num_rows):
                writer.writerow([f'row{i}col1', f'row{i}col2', f'row{i}col3'])
        
        result = count_csv_lines(str(csv_file))
        assert result == num_rows


class TestUpdateMarkets:
    """Test cases for update_markets function"""

    def test_update_markets_creates_file(self, tmp_path, monkeypatch):
        """Test that update_markets creates CSV file with correct structure"""
        csv_file = tmp_path / "test_markets.csv"
        
        # Mock the requests.get to avoid actual API calls
        class MockResponse:
            status_code = 200
            def json(self):
                return []  # Return empty list to stop iteration
        
        import requests
        def mock_get(*args, **kwargs):
            return MockResponse()
        
        monkeypatch.setattr(requests, "get", mock_get)
        
        update_markets(str(csv_file), batch_size=500)
        
        # Verify file was created
        assert csv_file.exists()
        
        # Verify headers
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            expected_headers = [
                'createdAt', 'id', 'question', 'answer1', 'answer2', 'neg_risk', 
                'market_slug', 'token1', 'token2', 'condition_id', 'volume', 'ticker', 'closedTime'
            ]
            assert headers == expected_headers

    def test_update_markets_appends_to_existing(self, tmp_path, monkeypatch):
        """Test that update_markets appends to existing file"""
        csv_file = tmp_path / "test_markets.csv"
        
        # Create existing file with one row
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['createdAt', 'id', 'question', 'answer1', 'answer2', 'neg_risk', 
                           'market_slug', 'token1', 'token2', 'condition_id', 'volume', 'ticker', 'closedTime'])
            writer.writerow(['2024-01-01', 'market1', 'Test?', 'Yes', 'No', False,
                           'test-slug', 'token1', 'token2', 'cond1', '1000', 'TEST', '2024-12-31'])
        
        # Mock the requests.get to return empty (no new markets)
        class MockResponse:
            status_code = 200
            def json(self):
                return []
        
        import requests
        def mock_get(*args, **kwargs):
            return MockResponse()
        
        monkeypatch.setattr(requests, "get", mock_get)
        
        update_markets(str(csv_file), batch_size=500)
        
        # Verify file still has the original row
        line_count = count_csv_lines(str(csv_file))
        assert line_count == 1  # Original row should still be there

    def test_update_markets_handles_rate_limit(self, tmp_path, monkeypatch):
        """Test that update_markets handles rate limiting correctly"""
        csv_file = tmp_path / "test_markets.csv"
        
        call_count = {'count': 0}
        
        class MockResponse:
            def __init__(self, call_num):
                self.call_num = call_num
                if call_num == 0:
                    self.status_code = 429  # Rate limited on first call
                else:
                    self.status_code = 200  # Success on subsequent calls
            
            def json(self):
                return []  # Return empty to stop iteration
        
        import requests
        import time
        
        def mock_get(*args, **kwargs):
            response = MockResponse(call_count['count'])
            call_count['count'] += 1
            return response
        
        def mock_sleep(seconds):
            pass  # Don't actually sleep in tests
        
        monkeypatch.setattr(requests, "get", mock_get)
        monkeypatch.setattr(time, "sleep", mock_sleep)
        
        update_markets(str(csv_file), batch_size=500)
        
        # Should have been called at least twice (once with 429, once with 200)
        assert call_count['count'] >= 2
        assert csv_file.exists()
