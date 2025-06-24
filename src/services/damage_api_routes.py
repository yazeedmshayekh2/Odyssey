from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
import jwt
from functools import wraps
import os
from datetime import datetime
from bson import ObjectId
from src.config.db import get_db

damage_api_bp = Blueprint('damage_api', __name__, url_prefix='/api/damage')

def token_required(f):
    """Decorator to require JWT token for protected routes"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        
        if auth_header:
            try:
                token = auth_header.split(" ")[1]  # Bearer <token>
            except IndexError:
                return jsonify({'message': 'Invalid token format', 'success': False}), 401
        
        if not token:
            return jsonify({'message': 'Authentication required', 'success': False}), 401
        
        try:
            # Use the same JWT secret as the auth module
            jwt_secret = os.getenv('JWT_SECRET', 'odyssey_secret_key_change_in_production')
            data = jwt.decode(token, jwt_secret, algorithms=['HS256'])
            current_user_id = data['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired', 'success': False}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token', 'success': False}), 401
        
        return f(current_user_id, *args, **kwargs)
    
    return decorated

@damage_api_bp.route('/reports', methods=['GET', 'OPTIONS'])
@cross_origin()
@token_required
def get_damage_reports(current_user_id):
    """Get all damage reports for the current user"""
    try:
        db = get_db()
        
        # Get query parameters
        car_id = request.args.get('carId')
        status = request.args.get('status')
        limit = request.args.get('limit', type=int)
        
        # Build query
        query = {'user_id': ObjectId(current_user_id)}
        if car_id:
            query['car_id'] = ObjectId(car_id)
        if status:
            query['status'] = status
        
        # Find reports
        reports_cursor = db.damage_reports.find(query).sort('created_at', -1)
        
        if limit:
            reports_cursor = reports_cursor.limit(limit)
        
        reports = []
        for report in reports_cursor:
            # Convert ObjectId to string for JSON serialization
            report['_id'] = str(report['_id'])
            report['user_id'] = str(report['user_id'])
            report['car_id'] = str(report['car_id']) if 'car_id' in report else None
            
            # Convert datetime to ISO string
            if 'created_at' in report:
                report['created_at'] = report['created_at'].isoformat() if isinstance(report['created_at'], datetime) else report['created_at']
            if 'updated_at' in report:
                report['updated_at'] = report['updated_at'].isoformat() if isinstance(report['updated_at'], datetime) else report['updated_at']
            
            reports.append(report)
        
        return jsonify({
            'success': True,
            'reports': reports,
            'count': len(reports)
        })
        
    except Exception as e:
        print(f"Error fetching damage reports: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to fetch damage reports',
            'error': str(e)
        }), 500

@damage_api_bp.route('/reports/<report_id>', methods=['GET', 'OPTIONS'])
@cross_origin()
@token_required
def get_damage_report(current_user_id, report_id):
    """Get a specific damage report by ID"""
    try:
        db = get_db()
        
        # Find the report
        report = db.damage_reports.find_one({
            '_id': ObjectId(report_id),
            'user_id': ObjectId(current_user_id)
        })
        
        if not report:
            return jsonify({
                'success': False,
                'message': 'Report not found'
            }), 404
        
        # Convert ObjectId to string for JSON serialization
        report['_id'] = str(report['_id'])
        report['user_id'] = str(report['user_id'])
        report['car_id'] = str(report['car_id']) if 'car_id' in report else None
        
        # Convert datetime to ISO string
        if 'created_at' in report:
            report['created_at'] = report['created_at'].isoformat() if isinstance(report['created_at'], datetime) else report['created_at']
        if 'updated_at' in report:
            report['updated_at'] = report['updated_at'].isoformat() if isinstance(report['updated_at'], datetime) else report['updated_at']
        
        return jsonify({
            'success': True,
            'report': report
        })
        
    except Exception as e:
        print(f"Error fetching damage report: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to fetch damage report',
            'error': str(e)
        }), 500

@damage_api_bp.route('/reports', methods=['POST'])
@cross_origin()
@token_required
def create_damage_report(current_user_id):
    """Create a new damage report"""
    try:
        db = get_db()
        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        # Create report document
        report = {
            'user_id': ObjectId(current_user_id),
            'car_id': ObjectId(data.get('car_id')) if data.get('car_id') else None,
            'car_info': data.get('car_info', {}),
            'damage_detected': data.get('damage_detected', False),
            'damage_percentage': data.get('damage_percentage', 0),
            'damage_types': data.get('damage_types', []),
            'confidence_scores': data.get('confidence_scores', {}),
            'images': data.get('images', {}),
            'status': data.get('status', 'processing'),
            'notes': data.get('notes', ''),
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        
        # Insert report
        result = db.damage_reports.insert_one(report)
        
        # Return created report
        report['_id'] = str(result.inserted_id)
        report['user_id'] = str(report['user_id'])
        report['car_id'] = str(report['car_id']) if report['car_id'] else None
        report['created_at'] = report['created_at'].isoformat()
        report['updated_at'] = report['updated_at'].isoformat()
        
        return jsonify({
            'success': True,
            'report': report,
            'message': 'Damage report created successfully'
        }), 201
        
    except Exception as e:
        print(f"Error creating damage report: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to create damage report',
            'error': str(e)
        }), 500

@damage_api_bp.route('/reports/<report_id>/status', methods=['PUT', 'OPTIONS'])
@cross_origin()
@token_required
def update_report_status(current_user_id, report_id):
    """Update damage report status"""
    try:
        db = get_db()
        data = request.get_json()
        
        if not data or 'status' not in data:
            return jsonify({
                'success': False,
                'message': 'Status is required'
            }), 400
        
        # Update the report
        result = db.damage_reports.update_one(
            {
                '_id': ObjectId(report_id),
                'user_id': ObjectId(current_user_id)
            },
            {
                '$set': {
                    'status': data['status'],
                    'updated_at': datetime.utcnow()
                }
            }
        )
        
        if result.matched_count == 0:
            return jsonify({
                'success': False,
                'message': 'Report not found'
            }), 404
        
        return jsonify({
            'success': True,
            'message': 'Report status updated successfully'
        })
        
    except Exception as e:
        print(f"Error updating report status: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to update report status',
            'error': str(e)
        }), 500

@damage_api_bp.route('/accident-reports', methods=['POST', 'OPTIONS'])
@cross_origin()
@token_required
def create_accident_report(current_user_id):
    """Create a new accident report"""
    try:
        db = get_db()
        
        # Handle both JSON and form data
        if request.content_type and 'application/json' in request.content_type:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        # Create accident report document
        accident_report = {
            'user_id': ObjectId(current_user_id),
            'car_id': ObjectId(data.get('carId')) if data.get('carId') else None,
            'incident_date': data.get('incidentDate'),
            'incident_time': data.get('incidentTime'),
            'location': data.get('location'),
            'description': data.get('description', ''),
            'damage_description': data.get('damageDescription', ''),
            'witnesses': data.get('witnesses', ''),
            'police_report': data.get('policeReport') == 'true' if isinstance(data.get('policeReport'), str) else bool(data.get('policeReport')),
            'police_report_number': data.get('policeReportNumber', ''),
            'status': 'submitted',
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        
        # Handle file uploads (images)
        if request.files:
            accident_report['images'] = []
            for file_key in request.files:
                file = request.files[file_key]
                if file and file.filename:
                    # In a real implementation, you'd save the file and store the path
                    # For now, just store the filename
                    accident_report['images'].append({
                        'filename': file.filename,
                        'uploaded_at': datetime.utcnow().isoformat()
                    })
        
        # Insert accident report
        result = db.accident_reports.insert_one(accident_report)
        
        # Return created report
        accident_report['_id'] = str(result.inserted_id)
        accident_report['user_id'] = str(accident_report['user_id'])
        accident_report['car_id'] = str(accident_report['car_id']) if accident_report['car_id'] else None
        accident_report['created_at'] = accident_report['created_at'].isoformat()
        accident_report['updated_at'] = accident_report['updated_at'].isoformat()
        
        return jsonify({
            'success': True,
            'report': accident_report,
            'reportId': str(result.inserted_id),
            'message': 'Accident report submitted successfully'
        }), 201
        
    except Exception as e:
        print(f"Error creating accident report: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to create accident report',
            'error': str(e)
        }), 500 